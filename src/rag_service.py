# -*- coding: utf-8 -*-
"""
RAG 法律检索增强引擎 (Legal RAG Retrieval Service)
===================================================
实现: 三路并行召回(语义/关键词/图谱) + BGE-Reranker融合重排序
"""
from __future__ import annotations
import enum, math, random, time
from dataclasses import dataclass, field

# ── 法律数据模型 ──
@dataclass
class Statute:
    """法条"""
    statute_id: str; title: str; article: str; full_text: str
    effect_level: int = 3  # 1宪法>2法律>3行政法规>4司法解释
    effective_from: str = "2021-01-01"; is_abolished: bool = False
    related_ids: list[str] = field(default_factory=list)

@dataclass
class Case:
    """判例"""
    case_id: str; case_number: str; court_level: int  # 1最高法>2高院>3中院>4基层
    cause_of_action: str; facts: str; judgment: str
    cited_statutes: list[str] = field(default_factory=list)

@dataclass
class RetrievalResult:
    """单条召回结果"""
    source: str  # milvus/es/neo4j
    item_id: str; item_type: str  # statute/case
    title: str; snippet: str
    raw_score: float; rerank_score: float = 0.0

# ── 模拟法律知识库 ──
STATUTE_DB: list[Statute] = [
    Statute("S001", "民法典", "第577条", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。", 2, related_ids=["S002","S003"]),
    Statute("S002", "民法典", "第584条", "当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，损失赔偿额应当相当于因违约所造成的损失。", 2, related_ids=["S001"]),
    Statute("S003", "民法典", "第585条", "当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。", 2, related_ids=["S001","S002"]),
    Statute("S004", "民法典", "第1043条", "家庭应当树立优良家风，弘扬家庭美德，重视家庭文明建设。", 2, related_ids=["S005"]),
    Statute("S005", "民法典", "第1046条", "结婚应当男女双方完全自愿，禁止任何一方对另一方加以强迫，禁止任何组织或者个人加以干涉。", 2),
    Statute("S006", "合同法", "第107条", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。", 2, is_abolished=True),
    Statute("S007", "著作权法", "第52条", "有下列侵权行为的，应当根据情况承担停止侵害、消除影响、赔礼道歉、赔偿损失等民事责任。", 2),
    Statute("S008", "反不正当竞争法", "第17条", "经营者违反本法规定，给他人造成损害的，应当依法承担民事责任。", 2),
]

CASE_DB: list[Case] = [
    Case("C001", "(2023)最高法民终456号", 1, "买卖合同纠纷", "原告与被告签订设备采购合同，被告逾期交付", "判决被告赔偿违约金", ["S001","S003"]),
    Case("C002", "(2024)京民终123号", 3, "服务合同纠纷", "原告委托被告开发软件系统，被告交付不合格", "判决被告退还部分费用并赔偿损失", ["S001","S002"]),
    Case("C003", "(2023)沪高民终789号", 2, "知识产权侵权", "原告诉被告侵犯软件著作权", "判决被告停止侵权并赔偿50万元", ["S007","S008"]),
]

# ── 模拟向量相似度 ──
def _cosine_sim(query: str, text: str) -> float:
    """简化版语义相似度 (生产中用 BGE-M3 Embedding + Milvus)"""
    q_words = set(query); t_words = set(text)
    intersection = q_words & t_words
    if not q_words or not t_words: return 0.0
    return len(intersection) / math.sqrt(len(q_words) * len(t_words))

# ── 三路召回服务 ──
class MilvusRetriever:
    """语义向量召回 (模拟 Milvus + BGE-M3, Top-K=50)"""
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        results = []
        for s in STATUTE_DB:
            if s.is_abolished: continue
            score = _cosine_sim(query, s.full_text + s.title + s.article)
            results.append(RetrievalResult("milvus", s.statute_id, "statute", f"{s.title}{s.article}", s.full_text[:80], score))
        for c in CASE_DB:
            score = _cosine_sim(query, c.facts + c.cause_of_action + c.judgment)
            results.append(RetrievalResult("milvus", c.case_id, "case", c.case_number, c.facts[:80], score))
        results.sort(key=lambda r: r.raw_score, reverse=True)
        return results[:top_k]

class ESRetriever:
    """关键词精确召回 (模拟 Elasticsearch BM25)"""
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        results = []
        keywords = [w for w in query if len(w.strip()) > 0]
        for s in STATUTE_DB:
            if s.is_abolished: continue
            match_count = sum(1 for k in keywords if k in s.full_text or k in s.title or k in s.article)
            if match_count > 0:
                score = match_count / len(keywords) if keywords else 0
                results.append(RetrievalResult("es", s.statute_id, "statute", f"{s.title}{s.article}", s.full_text[:80], score))
        for c in CASE_DB:
            match_count = sum(1 for k in keywords if k in c.facts or k in c.cause_of_action)
            if match_count > 0:
                score = match_count / len(keywords) if keywords else 0
                results.append(RetrievalResult("es", c.case_id, "case", c.case_number, c.facts[:80], score))
        results.sort(key=lambda r: r.raw_score, reverse=True)
        return results[:top_k]

class Neo4jRetriever:
    """知识图谱关联召回 (模拟 Neo4j 法条引用链遍历)"""
    def retrieve(self, statute_ids: list[str], depth: int = 2) -> list[RetrievalResult]:
        results = []; visited = set(statute_ids)
        def _traverse(sid, d):
            if d > depth: return
            for s in STATUTE_DB:
                if s.statute_id == sid:
                    for rid in s.related_ids:
                        if rid not in visited:
                            visited.add(rid)
                            for rs in STATUTE_DB:
                                if rs.statute_id == rid and not rs.is_abolished:
                                    results.append(RetrievalResult("neo4j", rs.statute_id, "statute",
                                        f"{rs.title}{rs.article}", f"关联法条(深度{d}): {rs.full_text[:60]}", 0.5/d))
                            _traverse(rid, d + 1)
        for sid in statute_ids: _traverse(sid, 1)
        return results

# ── 重排序 ──
class BGEReranker:
    """融合重排序 (模拟 BGE-Reranker-V2 Cross-Encoder)"""
    EFFECT_BOOST = {1: 0.15, 2: 0.10, 3: 0.05, 4: 0.0}  # 宪法>法律>法规>解释
    COURT_BOOST  = {1: 0.12, 2: 0.08, 3: 0.04, 4: 0.0}   # 最高法>高院>中院>基层

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 10) -> list[RetrievalResult]:
        for c in candidates:
            base = c.raw_score
            boost = 0.0
            if c.item_type == "statute":
                s = next((x for x in STATUTE_DB if x.statute_id == c.item_id), None)
                if s: boost += self.EFFECT_BOOST.get(s.effect_level, 0)
            elif c.item_type == "case":
                cs = next((x for x in CASE_DB if x.case_id == c.item_id), None)
                if cs: boost += self.COURT_BOOST.get(cs.court_level, 0)
            c.rerank_score = base + boost + random.uniform(0, 0.05)
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        seen = set(); unique = []
        for c in candidates:
            if c.item_id not in seen: seen.add(c.item_id); unique.append(c)
        return unique[:top_k]

# ── RAG 主服务 ──
class RAGService:
    """法律检索增强服务 — 三路召回 + Rerank"""
    def __init__(self):
        self.milvus = MilvusRetriever()
        self.es = ESRetriever()
        self.neo4j = Neo4jRetriever()
        self.reranker = BGEReranker()

    def hybrid_retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        print(f"\n🔍 RAG 检索: \"{query[:50]}...\"")
        t0 = time.time()
        # 三路并行召回 (demo中串行模拟)
        milvus_results = self.milvus.retrieve(query, top_k=5)
        print(f"   📡 Milvus 语义召回: {len(milvus_results)} 条")
        es_results = self.es.retrieve(query, top_k=5)
        print(f"   📡 ES 关键词召回:   {len(es_results)} 条")
        hit_ids = [r.item_id for r in milvus_results + es_results if r.item_type == "statute"]
        neo4j_results = self.neo4j.retrieve(hit_ids[:3])
        print(f"   📡 Neo4j 图谱召回:  {len(neo4j_results)} 条")
        # 融合重排序
        all_candidates = milvus_results + es_results + neo4j_results
        final = self.reranker.rerank(query, all_candidates, top_k)
        elapsed = (time.time() - t0) * 1000
        print(f"   ⏱️  总耗时: {elapsed:.0f}ms | 最终返回 Top-{len(final)}")
        return final

def demo():
    print("=" * 60); print("  RAG 法律检索增强引擎 — 可运行 Demo"); print("=" * 60)
    rag = RAGService()
    query = "被告未按期履行合同义务，需要承担什么违约责任？赔偿损失如何计算？"
    results = rag.hybrid_retrieve(query)
    print(f"\n📋 Top-{len(results)} 检索结果:")
    for i, r in enumerate(results, 1):
        src_icon = {"milvus": "🧠", "es": "🔑", "neo4j": "🕸️"}.get(r.source, "?")
        print(f"  {i}. {src_icon} [{r.source}] {r.title}")
        print(f"     类型: {r.item_type} | Rerank分: {r.rerank_score:.3f} | 摘要: {r.snippet[:50]}...")

if __name__ == "__main__":
    demo()
