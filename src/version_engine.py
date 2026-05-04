# -*- coding: utf-8 -*-
"""
文书版本控制引擎 (Document Version Engine)
==========================================
实现: Myers Diff + OT操作转换 + 三路合并 + Delta/Snapshot混合存储
"""
from __future__ import annotations
import enum, time, hashlib, json
from dataclasses import dataclass, field
from typing import Optional

# ── 操作类型 ──
class OpType(enum.Enum):
    INSERT = "insert"; DELETE = "delete"; RETAIN = "retain"

@dataclass
class Operation:
    op_type: OpType; position: int; content: str = ""; length: int = 0

@dataclass
class Delta:
    """版本差异 (操作序列)"""
    operations: list[Operation] = field(default_factory=list)
    from_version: int = 0; to_version: int = 0

@dataclass
class Version:
    version_id: int; content: str; timestamp: str
    delta_from_prev: Optional[Delta] = None; is_snapshot: bool = False
    author: str = "user"; content_hash: str = ""
    def __post_init__(self):
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]

# ── Myers Diff 算法 ──
class MyersDiff:
    """Myers差分算法 — 计算最小编辑操作序列"""
    def compute_diff(self, old: str, new: str) -> list[tuple[str, str]]:
        """返回 [(tag, text), ...] 其中 tag = 'equal'/'insert'/'delete'"""
        old_lines = old.split('\n'); new_lines = new.split('\n')
        m, n = len(old_lines), len(new_lines)
        # 简化 LCS 实现
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if old_lines[i-1]==new_lines[j-1] else max(dp[i-1][j], dp[i][j-1])
        # 回溯
        result = []; i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and old_lines[i-1] == new_lines[j-1]:
                result.append(("equal", old_lines[i-1])); i -= 1; j -= 1
            elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
                result.append(("insert", new_lines[j-1])); j -= 1
            else:
                result.append(("delete", old_lines[i-1])); i -= 1
        result.reverse()
        return result

    def compute_delta(self, old: str, new: str) -> Delta:
        """计算两版本间的Delta"""
        diffs = self.compute_diff(old, new)
        ops = []; pos = 0
        for tag, text in diffs:
            if tag == "equal": ops.append(Operation(OpType.RETAIN, pos, length=len(text))); pos += len(text)
            elif tag == "insert": ops.append(Operation(OpType.INSERT, pos, content=text))
            elif tag == "delete": ops.append(Operation(OpType.DELETE, pos, content=text, length=len(text))); pos += len(text)
        return Delta(operations=ops)

# ── OT 操作转换引擎 ──
class OTEngine:
    """OT引擎 — 处理并发编辑冲突"""
    def transform(self, op_a: Operation, op_b: Operation) -> tuple[Operation, Operation]:
        """转换两个并发操作，使其可按任意顺序应用"""
        a_prime, b_prime = Operation(op_a.op_type, op_a.position, op_a.content, op_a.length), \
                           Operation(op_b.op_type, op_b.position, op_b.content, op_b.length)
        if op_a.op_type == OpType.INSERT and op_b.op_type == OpType.INSERT:
            if op_a.position <= op_b.position: b_prime.position += len(op_a.content)
            else: a_prime.position += len(op_b.content)
        elif op_a.op_type == OpType.INSERT and op_b.op_type == OpType.DELETE:
            if op_a.position <= op_b.position: b_prime.position += len(op_a.content)
            elif op_a.position >= op_b.position + op_b.length: a_prime.position -= op_b.length
        elif op_a.op_type == OpType.DELETE and op_b.op_type == OpType.INSERT:
            if op_b.position <= op_a.position: a_prime.position += len(op_b.content)
            elif op_b.position >= op_a.position + op_a.length: b_prime.position -= op_a.length
        return a_prime, b_prime

# ── 三路合并 ──
class ThreeWayMerge:
    """三路合并引擎 — 基于LCA的合并"""
    def __init__(self): self.diff = MyersDiff()

    def merge(self, base: str, version_a: str, version_b: str) -> tuple[str, list[dict]]:
        """三路合并，返回 (merged_text, conflicts)"""
        diff_a = self.diff.compute_diff(base, version_a)
        diff_b = self.diff.compute_diff(base, version_b)
        merged_lines, conflicts = [], []
        ia, ib = 0, 0
        while ia < len(diff_a) or ib < len(diff_b):
            a = diff_a[ia] if ia < len(diff_a) else None
            b = diff_b[ib] if ib < len(diff_b) else None
            if a and b and a[0] == "equal" and b[0] == "equal" and a[1] == b[1]:
                merged_lines.append(a[1]); ia += 1; ib += 1
            elif a and a[0] == "insert" and (not b or b[0] != "insert"):
                merged_lines.append(a[1]); ia += 1
            elif b and b[0] == "insert" and (not a or a[0] != "insert"):
                merged_lines.append(b[1]); ib += 1
            elif a and b and a[0] == "insert" and b[0] == "insert" and a[1] != b[1]:
                conflicts.append({"type": "both_insert", "line": len(merged_lines), "a": a[1], "b": b[1]})
                merged_lines.append(f"<<<<<<< A: {a[1]}"); merged_lines.append(f"======= B: {b[1]}"); merged_lines.append(">>>>>>>")
                ia += 1; ib += 1
            elif a and a[0] == "delete":
                ia += 1; ib = min(ib + 1, len(diff_b))
            elif b and b[0] == "delete":
                ib += 1; ia = min(ia + 1, len(diff_a))
            else:
                if a: merged_lines.append(a[1]); ia += 1
                if b and (not a or b[1] != a[1]): merged_lines.append(b[1]); ib += 1
                if not a and not b: break
        return "\n".join(merged_lines), conflicts

# ── 版本控制主引擎 ──
SNAPSHOT_INTERVAL = 10

class VersionEngine:
    """文书版本控制引擎 — Delta链 + Snapshot + OT"""
    def __init__(self):
        self.diff_algo = MyersDiff(); self.ot = OTEngine(); self.merge = ThreeWayMerge()
        self.versions: list[Version] = []

    def init_document(self, content: str, author: str = "system") -> Version:
        v = Version(0, content, time.strftime("%Y-%m-%d %H:%M:%S"), is_snapshot=True, author=author)
        self.versions.append(v); return v

    def commit(self, new_content: str, author: str = "user") -> Version:
        if not self.versions: return self.init_document(new_content, author)
        prev = self.versions[-1]; vid = prev.version_id + 1
        delta = self.diff_algo.compute_delta(prev.content, new_content)
        delta.from_version = prev.version_id; delta.to_version = vid
        is_snap = (vid % SNAPSHOT_INTERVAL == 0)
        v = Version(vid, new_content, time.strftime("%Y-%m-%d %H:%M:%S"), delta, is_snap, author)
        self.versions.append(v)
        return v

    def get_version(self, version_id: int) -> Optional[Version]:
        return next((v for v in self.versions if v.version_id == version_id), None)

    def compare(self, v1_id: int, v2_id: int) -> list[tuple[str, str]]:
        v1, v2 = self.get_version(v1_id), self.get_version(v2_id)
        if not v1 or not v2: raise ValueError("版本不存在")
        return self.diff_algo.compute_diff(v1.content, v2.content)

    def merge_versions(self, base_id: int, a_id: int, b_id: int) -> tuple[str, list[dict]]:
        base, va, vb = self.get_version(base_id), self.get_version(a_id), self.get_version(b_id)
        if not all([base, va, vb]): raise ValueError("版本不存在")
        return self.merge.merge(base.content, va.content, vb.content)

    def get_history(self) -> list[dict]:
        return [{"id": v.version_id, "time": v.timestamp, "author": v.author,
                 "hash": v.content_hash, "snapshot": v.is_snapshot,
                 "ops": len(v.delta_from_prev.operations) if v.delta_from_prev else 0} for v in self.versions]

def demo():
    print("=" * 60); print("  文书版本控制引擎 — 可运行 Demo"); print("=" * 60)
    engine = VersionEngine()
    # 初始版本
    v0_text = "第一条 甲方应按时交付货物\n第二条 乙方应按时支付货款\n第三条 违约方承担违约责任"
    engine.init_document(v0_text, "律师A")
    print(f"\n📝 V0 (初始):\n{v0_text}")
    # 版本1: 律师A修改
    v1_text = "第一条 甲方应按时交付货物\n第二条 乙方应在收到货物后30日内支付货款\n第三条 违约方承担违约责任\n第四条 争议提交北京仲裁委员会仲裁"
    v1 = engine.commit(v1_text, "律师A")
    print(f"\n📝 V1 (律师A修改): {len(v1.delta_from_prev.operations)} 个操作")
    # 版本2: 律师B修改
    v2_text = "第一条 甲方应按时交付符合质量标准的货物\n第二条 乙方应按时支付货款\n第三条 违约方承担违约责任，违约金为合同金额的20%\n第五条 本合同一式两份"
    v2 = engine.commit(v2_text, "律师B")
    print(f"📝 V2 (律师B修改): {len(v2.delta_from_prev.operations)} 个操作")
    # 差异对比
    print(f"\n📊 V0 → V2 差异:")
    diffs = engine.compare(0, 2)
    for tag, text in diffs:
        icon = {"equal": "  ", "insert": "🟢", "delete": "🔴"}.get(tag, "?")
        print(f"  {icon} [{tag:6s}] {text}")
    # 三路合并
    print(f"\n🔀 三路合并 (Base=V0, A=V1, B=V2):")
    merged, conflicts = engine.merge_versions(0, 1, 2)
    print(f"  合并结果:\n{merged}")
    if conflicts: print(f"  ⚠️  冲突: {len(conflicts)} 处"); [print(f"    {c}") for c in conflicts]
    else: print(f"  ✅ 无冲突")
    # 版本历史
    print(f"\n📋 版本历史:")
    for h in engine.get_history():
        snap = " [SNAPSHOT]" if h["snapshot"] else ""
        print(f"  V{h['id']} | {h['time']} | {h['author']} | hash={h['hash']} | ops={h['ops']}{snap}")

if __name__ == "__main__":
    demo()
