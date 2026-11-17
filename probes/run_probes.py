#!/usr/bin/env python3
"""验收探针：pickle 样本批量风险分级（10 条契约）。

样本全部现场构造，不落盘、不执行。基线（未实现 `fickling.audit`）10 条全 FAIL。
"""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = []


def probe(name):
    def deco(fn):
        try:
            fn()
        except AssertionError as exc:
            RESULTS.append((name, False, str(exc) or "断言失败"))
        except Exception as exc:
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        else:
            RESULTS.append((name, True, ""))
        return fn

    return deco


def scan():
    from fickling.audit import scan_pickles

    return scan_pickles


SAFE = pickle.dumps({"user": "alice", "roles": ["reader"]}, protocol=0)
OS_CMD = b"cos\nsystem\n(S'echo pwned'\ntR."
CODE_EVAL = b"c__builtin__\neval\n(S'1+1'\ntR."
SUBPROCESS = b"csubprocess\ncheck_output\n(S'id'\ntR."
NUMPY = b"cnumpy\ncore\n(S'x'\ntR."
TRUNCATED = b"cos\nsys"


@probe("P01 批量入口与安全样本")
def p01():
    fn = scan()
    out = fn([SAFE])
    assert hasattr(out, "items") and hasattr(out, "stats"), "需返回 items 与 stats"
    item = out.items[0]
    for key in ("index", "ok", "verdict", "reasons", "severity", "digest"):
        assert key in item, f"item 缺字段 {key}"
    assert item["verdict"] == "safe", f"纯数据 pickle 应 safe，实际 {item['verdict']}"
    assert item["ok"] and item["severity"] == "none", f"safe 样本 severity 应为 none，实际 {item['severity']}"


@probe("P02 os.system 判恶意")
def p02():
    fn = scan()
    item = fn([OS_CMD]).items[0]
    assert item["verdict"] == "malicious", f"应判 malicious，实际 {item['verdict']}"
    assert "os_command" in item["reasons"], f"应带 os_command，实际 {item['reasons']}"
    assert item["dangerous"] and not item["dangerous_unknown"], "os 作为模块应在已知危险名单内"


@probe("P03 代码执行类判恶意")
def p03():
    fn = scan()
    item = fn([CODE_EVAL]).items[0]
    assert item["verdict"] == "malicious", f"eval 应判 malicious，实际 {item['verdict']}"
    assert "code_exec" in item["reasons"], f"应带 code_exec，实际 {item['reasons']}"


@probe("P04 子进程调用判恶意")
def p04():
    fn = scan()
    item = fn([SUBPROCESS]).items[0]
    assert item["verdict"] == "malicious", f"subprocess 应判 malicious，实际 {item['verdict']}"
    assert "subprocess" in item["reasons"], f"应带 subprocess，实际 {item['reasons']}"


@probe("P05 截断样本隔离")
def p05():
    fn = scan()
    out = fn([TRUNCATED, SAFE])
    assert out.items[0]["verdict"] == "malformed", f"截断应记 malformed，实际 {out.items[0]['verdict']}"
    assert not out.items[0]["ok"], "malformed 的 ok 应为假"
    assert out.items[1]["verdict"] == "safe", "坏样本之后的正常样本应继续"


@probe("P06 重复样本识别")
def p06():
    fn = scan()
    out = fn([SAFE, SAFE])
    assert out.items[0]["digest"] == out.items[1]["digest"], "同内容 digest 应一致"
    assert out.items[1].get("duplicate_of") == 0, f"第二个应记 duplicate_of=0，实际 {out.items[1].get('duplicate_of')}"


@probe("P07 白名单降级为可疑")
def p07():
    fn = scan()
    policy = {"allow_globals": ["numpy.core"]}
    item = fn([NUMPY], policy=policy).items[0]
    assert item["verdict"] == "suspicious", f"白名单内的 GLOBAL 应为 suspicious，实际 {item['verdict']}"
    assert item["severity"] == "low", f"应为 low，实际 {item['severity']}"
    plain = fn([NUMPY]).items[0]
    assert plain["verdict"] == "malicious", f"不在白名单时应 malicious，实际 {plain['verdict']}"


@probe("P08 混合批次与 ok 语义")
def p08():
    fn = scan()
    out = fn([SAFE, OS_CMD, TRUNCATED])
    verdicts = [i["verdict"] for i in out.items]
    assert verdicts == ["safe", "malicious", "malformed"], f"判定顺序不对：{verdicts}"
    okflags = [i["ok"] for i in out.items]
    assert okflags == [True, False, False], f"ok 语义不对：{okflags}"


@probe("P09 统计口径与空输入")
def p09():
    fn = scan()
    out = fn([SAFE, SAFE, OS_CMD, CODE_EVAL, TRUNCATED])
    st = out.stats
    assert st["safe"] == 2 and st["malicious"] == 2 and st["malformed"] == 1, f"计数不对：{st}"
    assert st["unique"] == 4, f"去重后应为 4，实际 {st['unique']}"
    assert st["by_reason"].get("os_command") == 1, f"by_reason 不对：{st['by_reason']}"
    empty = fn([])
    assert empty.items == [] and empty.stats["safe"] == 0, "空输入应返回空结构"


@probe("P10 批量性能与既有模块可用")
def p10():
    fn = scan()
    batch = [SAFE, OS_CMD, CODE_EVAL, SUBPROCESS] * 1250
    t0 = time.perf_counter()
    out = fn(batch)
    cost = time.perf_counter() - t0
    assert len(out.items) == 5000, "5000 条应全部产出"
    assert cost < 3.0, f"5000 条耗时 {cost:.2f}s 超过 3s"
    import fickling

    assert hasattr(fickling, "__version__") or hasattr(fickling, "fickle"), "既有 fickling 包被破坏"


def main():
    print("=" * 60)
    print("验收探针：pickle 样本批量风险分级（10 条契约）")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"PASS  {name}" if ok else f"FAIL  {name}\n      {detail}")
    print("-" * 60)
    print(f"探针结果：{passed}/{len(RESULTS)} 通过")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
