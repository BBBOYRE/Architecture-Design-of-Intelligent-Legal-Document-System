# -*- coding: utf-8 -*-
"""
引文核验模块 (Citation Verification Guard)
==========================================
实现: 法条/判例引用溯源 + 精确/模糊匹配 + 幻觉分级阻断
"""
from __future__ import annotations
import enum, re
from dataclasses import dataclass, field

class HallucinationLevel(enum.Enum):
    H0_FATAL = 0    # 伪造法条 → 阻断
    H1_SEVERE = 1   # 虚构判例 → 阻断
    H2_MODERATE = 2 # 编号偏移 → 警告
    H3_MINOR = 3    # 措辞问题 → 通过

class VerifyStatus(enum.Enum):
    VERIFIED = "verified"; CORRECTED = "corrected"; BLOCKED = "blocked"

@dataclass
class Citation:
    raw_text: str; citation_type: str  # statute / case
    statute_id: str = ""; article: str = ""
    case_number: str = ""; verified: bool = False
    status: VerifyStatus = VerifyStatus.BLOCKED; correction: str = ""

@dataclass
class RAGHit:
    item_id: str; item_type: str; title: str; full_text: str; article: str = ""

@dataclass
class VerifyResult:
    original_text: str; verified_text: str; citations_total: int
    verified_count: int; corrected_count: int; blocked_count: int
    details: list[Citation] = field(default_factory=list)
    passed: bool = True

# ── 模拟 RAG 命中库 (生产中从 RAG 服务获取) ──
MOCK_RAG_HITS = [
    RAGHit("S001", "statute", "民法典", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。", "第577条"),
    RAGHit("S002", "statute", "民法典", "损失赔偿额应当相当于因违约所造成的损失。", "第584条"),
    RAGHit("S003", "statute", "民法典", "当事人可以约定违约金。", "第585条"),
    RAGHit("C001", "case", "(2023)最高法民终456号", "判决被告赔偿违约金"),
    RAGHit("C002", "case", "(2024)京民终123号", "判决被告退还部分费用"),
]

class CitationExtractor:
    """从生成文本中提取引用"""
    STATUTE_PAT = re.compile(r'《([\u4e00-\u9fa5]+)》第(\d+)条')
    CASE_PAT = re.compile(r'[\(（](\d{4})[\)）]([\u4e00-\u9fa5]+\d+号)')

    def extract(self, text: str) -> list[Citation]:
        citations = []
        for m in self.STATUTE_PAT.finditer(text):
            citations.append(Citation(m.group(), "statute", article=f"第{m.group(2)}条"))
        for m in self.CASE_PAT.finditer(text):
            citations.append(Citation(m.group(), "case", case_number=f"({m.group(1)}){m.group(2)}"))
        return citations

class HallucinationGuard:
    """
    引文核验Guard — 法律LLM幻觉防控核心模块
    
    核心原则：所有法条/判例引用必须100%可溯源到RAG命中
    """
    def __init__(self):
        self.extractor = CitationExtractor()
        self.hallucination_log: list[dict] = []

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        if not s1 or not s2: return 0.0
        len1, len2 = len(s1), len(s2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(len1+1): dp[i][0] = i
        for j in range(len2+1): dp[0][j] = j
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                dp[i][j] = dp[i-1][j-1] if s1[i-1]==s2[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return 1 - dp[len1][len2] / max(len1, len2)

    def _exact_match(self, citation: Citation, rag_hits: list[RAGHit]) -> RAGHit | None:
        for hit in rag_hits:
            if citation.citation_type == "statute" and hit.item_type == "statute":
                if citation.article and hit.article and citation.article == hit.article:
                    if citation.raw_text and hit.title in citation.raw_text: return hit
            elif citation.citation_type == "case" and hit.item_type == "case":
                if citation.case_number and citation.case_number in hit.title: return hit
        return None

    def _fuzzy_match(self, citation: Citation, rag_hits: list[RAGHit], threshold: float = 0.85) -> RAGHit | None:
        best_hit, best_score = None, 0.0
        for hit in rag_hits:
            if citation.citation_type != hit.item_type: continue
            score = self._levenshtein_ratio(citation.raw_text, hit.title + hit.article)
            if score > best_score: best_score = score; best_hit = hit
        return best_hit if best_score >= threshold else None

    def verify_citations(self, generated_text: str, rag_hits: list[RAGHit] = None) -> VerifyResult:
        """核验生成文本中的所有引用"""
        if rag_hits is None: rag_hits = MOCK_RAG_HITS
        citations = self.extractor.extract(generated_text)
        verified_text = generated_text
        verified, corrected, blocked = 0, 0, 0

        for c in citations:
            hit = self._exact_match(c, rag_hits)
            if hit:
                c.verified = True; c.status = VerifyStatus.VERIFIED; verified += 1; continue
            hit = self._fuzzy_match(c, rag_hits)
            if hit:
                canonical = f"《{hit.title}》{hit.article}" if hit.item_type == "statute" else hit.title
                c.verified = True; c.status = VerifyStatus.CORRECTED; c.correction = canonical
                verified_text = verified_text.replace(c.raw_text, canonical); corrected += 1; continue
            c.status = VerifyStatus.BLOCKED; blocked += 1
            self.hallucination_log.append({"citation": c.raw_text, "type": c.citation_type,
                "level": HallucinationLevel.H0_FATAL.name if c.citation_type == "statute" else HallucinationLevel.H1_SEVERE.name})

        passed = blocked == 0
        return VerifyResult(generated_text, verified_text, len(citations), verified, corrected, blocked, citations, passed)

def demo():
    print("=" * 60); print("  引文核验模块 — 可运行 Demo"); print("=" * 60)
    guard = HallucinationGuard()

    # 测试1: 正常引用
    text1 = "根据《民法典》第577条的规定，当事人一方不履行合同义务应当承担违约责任。参见（2023）最高法民终456号判例。"
    print(f"\n📄 测试1 (正常引用):\n   {text1}")
    r1 = guard.verify_citations(text1)
    print(f"   ✅ 通过: {r1.passed} | 验证{r1.verified_count} 纠正{r1.corrected_count} 阻断{r1.blocked_count}")

    # 测试2: 含虚构法条
    text2 = "依据《民法典》第577条和《民法典》第999条（虚构），被告应赔偿。另见（2025）沪高民终999号（虚构判例）。"
    print(f"\n📄 测试2 (含虚构引用):\n   {text2}")
    r2 = guard.verify_citations(text2)
    print(f"   ❌ 通过: {r2.passed} | 验证{r2.verified_count} 纠正{r2.corrected_count} 阻断{r2.blocked_count}")
    for c in r2.details:
        icon = "✅" if c.status == VerifyStatus.VERIFIED else "🔄" if c.status == VerifyStatus.CORRECTED else "🚫"
        print(f"      {icon} {c.raw_text} → {c.status.value}" + (f" → {c.correction}" if c.correction else ""))

    # 测试3: 需纠正的引用
    text3 = "根据《民法典》第584条，损失赔偿额应当相当于因违约所造成的损失。"
    print(f"\n📄 测试3 (可验证引用):\n   {text3}")
    r3 = guard.verify_citations(text3)
    print(f"   ✅ 通过: {r3.passed} | 验证{r3.verified_count}")

    print(f"\n📋 幻觉日志: {len(guard.hallucination_log)} 条记录")
    for h in guard.hallucination_log: print(f"   🚨 {h}")

if __name__ == "__main__":
    demo()
