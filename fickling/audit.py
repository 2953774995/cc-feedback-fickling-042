"""Static risk triage for batches of pickle samples (audit entry point).

This module builds batch-processing capabilities on top of fickling's existing
purely static opcode parsing. It never runs a real deserialization such as
``pickle.loads`` against any sample; it only parses opcodes and call chains.

Public entry point: :func:`scan_pickles`.

Verdict semantics:

* ``safe``       -- pure data, or calls limited to known-safe builtin constructors;
* ``suspicious`` -- allowlisted third-party calls, global references that are never
  actually invoked, and similar low-confidence signals;
* ``malicious``  -- a dangerous call is reached (command execution / code execution /
  subprocess / any other known or unknown dangerous global);
* ``malformed``  -- truncated input, illegal opcode, interpretation failure, etc.
  A sample that cannot be parsed is isolated from the risk verdicts.

Reason codes (``reasons``; see the ``*_REASON`` constants) are stable and safe for
downstream aggregation. ``stats["by_reason"]`` tallies how often each code occurs.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fickling.fickle import (
    BUILTIN_MODULE_NAMES,
    SAFE_BUILTINS,
    UNSAFE_IMPORTS,
    Interpreter,
    Pickled,
)

# --------------------------------------------------------------------------- #
# Reason codes -- kept stable for stats["by_reason"] and downstream alerting.
# --------------------------------------------------------------------------- #
REASON_OS_COMMAND = "os_command"
REASON_CODE_EXEC = "code_exec"
REASON_SUBPROCESS = "subprocess"
REASON_DANGEROUS_IMPORT = "dangerous_import"
REASON_DANGEROUS_BUILTIN = "dangerous_builtin"
REASON_UNKNOWN_GLOBAL = "unknown_global"
REASON_ALLOWLISTED = "allowlisted_global"
REASON_SAFE_BUILTIN = "safe_builtin"
REASON_GLOBAL_REFERENCE = "global_reference"
REASON_PARSE_ERROR = "parse_error"
REASON_INVALID_OPCODE = "invalid_opcode"

# Verdict values.
SAFE = "safe"
SUSPICIOUS = "suspicious"
MALICIOUS = "malicious"
MALFORMED = "malformed"

# Severity values.
SEV_NONE = "none"
SEV_LOW = "low"
SEV_MEDIUM = "medium"
SEV_HIGH = "high"

# Command-execution family. posix/nt are the underlying implementations of os;
# popen/system share the same shell execution surface.
OS_MODULES: frozenset[str] = frozenset({"os", "posix", "nt"})

# Subprocess family, including private implementation modules and legacy aliases.
SUBPROCESS_MODULES: frozenset[str] = frozenset(
    {"subprocess", "_posixsubprocess", "commands", "multiprocessing", "pty"}
)

# Modules whose core capability is executing or compiling arbitrary code.
CODE_EXEC_MODULES: frozenset[str] = frozenset(
    {"code", "codeop", "runpy", "compile", "exec", "eval", "marshal", "doctest"}
)

# Builtin names that are themselves direct code-execution primitives.
CODE_EXEC_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile"})


@dataclass(frozen=True)
class GlobalTarget:
    """A global callable that is actually invoked (the module/attr from find_class)."""

    module: str
    attr: str

    @property
    def dotted(self) -> str:
        return f"{self.module}.{self.attr}"


@dataclass
class ScanReport:
    """Return value of :func:`scan_pickles`."""

    items: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _normalize_allowlist(policy: dict[str, Any] | None) -> tuple[str, ...]:
    """Normalize ``policy["allow_globals"]`` into de-duplicated, dot-trimmed, lowercase prefixes."""
    if not policy:
        return ()
    raw = policy.get("allow_globals") or ()
    seen: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        norm = entry.strip().strip(".").lower()
        if norm and norm not in seen:
            seen.append(norm)
    return tuple(seen)


def _is_allowlisted(target: GlobalTarget, allow_prefixes: Sequence[str]) -> bool:
    """Return whether the target matches an allowlist entry exactly or as a dotted subpath.

    For example, ``numpy.core`` matches ``numpy.core`` and ``numpy.core.multiarray`` but
    not ``numpy.corex``. Comparison is segment-aware to avoid naive prefix overmatching.
    """
    module = target.module.lower()
    full = target.dotted.lower()
    for prefix in allow_prefixes:
        if module == prefix or full == prefix:
            return True
        if module.startswith(prefix + ".") or full.startswith(prefix + "."):
            return True
    return False


def _collect_called_globals(tree: ast.AST) -> list[GlobalTarget]:
    """Find the globals that are *actually called* in the interpreted AST.

    The interpreter lowers ``GLOBAL`` / ``STACK_GLOBAL`` to an ``ImportFrom`` whose bound
    name becomes the root ``Name`` of a call. Only bindings that appear at the root of a
    ``Call`` are counted; an import that is never invoked (which has no side effect when
    unpickling) is handled separately as a suspicious reference so plain references are
    not mislabeled as malicious.
    """
    imported: dict[str, list[tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported.setdefault(alias.asname or alias.name, []).append(
                    (node.module, alias.name)
                )

    called_roots: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct call: system(...) / eval(...)
        if isinstance(func, ast.Name):
            called_roots.add(func.id)
        # Attribute call: object.__new__(...) -- take the root object name.
        elif isinstance(func, ast.Attribute):
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                called_roots.add(root.id)

    targets: list[GlobalTarget] = []
    seen: set[tuple[str, str]] = set()
    for root in called_roots:
        for module, attr in imported.get(root, ()):
            key = (module, attr)
            if key not in seen:
                seen.add(key)
                targets.append(GlobalTarget(module=module, attr=attr))
    return targets


def _collect_referenced_globals(tree: ast.AST) -> list[GlobalTarget]:
    """Return every ImportFrom binding, including references that are never invoked."""
    targets: list[GlobalTarget] = []
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                key = (node.module, alias.name)
                if key not in seen:
                    seen.add(key)
                    targets.append(GlobalTarget(module=node.module, attr=alias.name))
    return targets


def _classify_target(
    target: GlobalTarget,
    allow_prefixes: Sequence[str],
) -> tuple[str, str, bool]:
    """Classify one called global as ``(reason, severity, is_known_dangerous)``.

    The allowlist takes precedence: a match is downgraded to suspicious and never
    participates in malicious classification.
    """
    module = target.module
    module_root = module.partition(".")[0]

    if _is_allowlisted(target, allow_prefixes):
        return REASON_ALLOWLISTED, SEV_LOW, False

    # 1) Subprocess family.
    if module_root in SUBPROCESS_MODULES or module in SUBPROCESS_MODULES:
        return REASON_SUBPROCESS, SEV_HIGH, True

    # 2) Builtin code execution / dangerous builtins.
    if module_root in BUILTIN_MODULE_NAMES:
        if target.attr in CODE_EXEC_BUILTINS:
            return REASON_CODE_EXEC, SEV_HIGH, True
        if target.attr in SAFE_BUILTINS:
            # Known-safe builtin constructors (int/list/...): no risk.
            return REASON_SAFE_BUILTIN, SEV_NONE, False
        # Any other builtin (open/getattr/__import__/...): dangerous builtin.
        return REASON_DANGEROUS_BUILTIN, SEV_HIGH, True

    # 3) os / posix / nt command-execution surface.
    if module_root in OS_MODULES or module in OS_MODULES:
        return REASON_OS_COMMAND, SEV_HIGH, True

    # 4) Code-execution module family.
    if module_root in CODE_EXEC_MODULES or module in CODE_EXEC_MODULES:
        return REASON_CODE_EXEC, SEV_HIGH, True

    # 5) fickling's existing set of dangerous modules.
    if module_root in UNSAFE_IMPORTS or module in UNSAFE_IMPORTS:
        return REASON_DANGEROUS_IMPORT, SEV_HIGH, True

    # 6) Any other global that is actually called: treat as an unknown dangerous global.
    return REASON_UNKNOWN_GLOBAL, SEV_HIGH, False


def _audit_one(data: bytes, allow_prefixes: Sequence[str]) -> dict[str, Any]:
    """Statically parse and triage a single sample; any parse failure becomes malformed and never raises."""
    digest = hashlib.sha256(data).hexdigest()
    item: dict[str, Any] = {
        "index": -1,
        "ok": False,
        "verdict": MALFORMED,
        "severity": SEV_MEDIUM,
        "reasons": [],
        "digest": digest,
        "dangerous": [],
        "dangerous_unknown": [],
    }

    try:
        pickled = Pickled.load(data)
        if pickled.has_invalid_opcode:
            item["reasons"] = [REASON_INVALID_OPCODE]
            return item
        tree = Interpreter.interpret(pickled)
    except Exception as exc:
        # Truncated / illegal opcode / interpretation failure: isolate this sample.
        item["reasons"] = [REASON_PARSE_ERROR]
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item

    called = _collect_called_globals(tree)

    # No called global: either pure data (safe) or only un-invoked global references (suspicious).
    if not called:
        referenced = _collect_referenced_globals(tree)
        if not referenced:
            item["verdict"] = SAFE
            item["severity"] = SEV_NONE
            item["ok"] = True
            return item
        # Imported but never called: only a module import is produced, so keep it suspicious.
        item["verdict"] = SUSPICIOUS
        item["severity"] = SEV_LOW
        item["ok"] = True
        item["reasons"] = [REASON_GLOBAL_REFERENCE]
        item["dangerous_unknown"] = sorted({t.dotted for t in referenced})
        return item

    reasons: list[str] = []
    dangerous: list[str] = []
    unknown: list[str] = []
    allowlisted: list[str] = []
    worst_severity = SEV_NONE
    is_malicious = False

    for target in called:
        reason, severity, known = _classify_target(target, allow_prefixes)
        if reason == REASON_SAFE_BUILTIN:
            # Known-safe builtin constructor: excluded from every risk list.
            continue
        if reason not in reasons:
            reasons.append(reason)
        if reason == REASON_ALLOWLISTED:
            allowlisted.append(target.dotted)
            if severity == SEV_HIGH:
                is_malicious = True
        else:
            is_malicious = True
            if known:
                dangerous.append(target.dotted)
            else:
                unknown.append(target.dotted)
        if severity == SEV_HIGH:
            worst_severity = SEV_HIGH

    # No malicious target: an allowlist-only call is suspicious; safe builtins alone (e.g. int(...)) are safe.
    if not is_malicious:
        if allowlisted:
            item["verdict"] = SUSPICIOUS
            item["severity"] = SEV_LOW
            item["ok"] = True
            item["reasons"] = reasons or [REASON_ALLOWLISTED]
            item["dangerous"] = sorted(set(allowlisted))
            return item
        item["verdict"] = SAFE
        item["severity"] = SEV_NONE
        item["ok"] = True
        item["reasons"] = reasons
        return item

    item["verdict"] = MALICIOUS
    item["severity"] = worst_severity if worst_severity != SEV_NONE else SEV_HIGH
    item["ok"] = False
    item["reasons"] = reasons
    item["dangerous"] = sorted(set(dangerous))
    item["dangerous_unknown"] = sorted(set(unknown))
    return item


def _empty_stats() -> dict[str, Any]:
    return {
        SAFE: 0,
        SUSPICIOUS: 0,
        MALICIOUS: 0,
        MALFORMED: 0,
        "unique": 0,
        "total": 0,
        "duplicates": 0,
        "by_reason": {},
    }


def scan_pickles(
    samples: Iterable[bytes],
    policy: dict[str, Any] | None = None,
) -> ScanReport:
    """Statically triage a batch of pickle byte strings.

    :param samples: an iterable of pickle ``bytes``. They are never passed to ``pickle.loads``.
    :param policy: optional policy. Currently supports
        ``{"allow_globals": ["numpy.core", ...]}``; a global call matching the allowlist is
        downgraded to ``suspicious`` (``severity="low"``) instead of malicious.
    :returns: a :class:`ScanReport` with per-sample ``items`` and aggregate ``stats``.
        A failure to parse one sample only affects that sample (recorded as ``malformed``);
        subsequent samples keep being processed.
    """
    allow_prefixes = _normalize_allowlist(policy)
    stats = _empty_stats()
    items: list[dict[str, Any]] = []

    # digest -> index of first occurrence, used for de-duplication.
    first_seen: dict[str, int] = {}

    for index, raw in enumerate(samples):
        data = bytes(raw)
        item = _audit_one(data, allow_prefixes)
        item["index"] = index

        digest = item["digest"]
        if digest in first_seen:
            item["duplicate_of"] = first_seen[digest]
            stats["duplicates"] += 1
        else:
            first_seen[digest] = index
            stats["unique"] += 1

        items.append(item)
        stats["total"] += 1
        stats[item["verdict"]] += 1
        for reason in item["reasons"]:
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1

    return ScanReport(items=items, stats=stats)
