# -*- coding: utf-8 -*-
"""
合同审查模块 (Contract Review Module — UC9)
=============================================
实现: 条款分类(14类) + 规则引擎(400+规则模拟) + 风险评分 + 修改建议
"""
from __future__ import annotations
import enum, re, json
from dataclasses import dataclass, field

class ClauseType(enum.Enum):
    SUBJECT = "主体条款"; OBJECT = "标的条款"; PRICE = "价款条款"
    PERFORMANCE = "履行条款"; QUALITY = "质量条款"; BREACH = "违约条款"
    DISPUTE = "争议解决"; JURISDICTION = "管辖条款"; CONFIDENTIAL = "保密条款"
    FORCE_MAJEURE = "不可抗力"; NOTICE = "通知条款"; AMENDMENT = "变更条款"
    TERMINATION = "终止条款"; OTHER = "其他条款"

class RiskLevel(enum.Enum):
    HIGH = "高风险"; MEDIUM = "中风险"; LOW = "低风险"; NONE = "无风险"

@dataclass
class Clause:
    clause_id: str; clause_type: ClauseType; text: str; order_index: int
    risk_level: RiskLevel = RiskLevel.NONE
    risks: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

@dataclass
class ReviewReport:
    contract_type: str; perspective: str; total_clauses: int
    high_risk: int; medium_risk: int; low_risk: int
    compliance_score: int; clauses: list[Clause] = field(default_factory=list)

# ── 条款分类器 (模拟 Legal-BERT) ──
CLAUSE_KEYWORDS = {
    ClauseType.SUBJECT: ["甲方", "乙方", "丙方", "委托方", "受托方", "出卖人", "买受人"],
    ClauseType.OBJECT: ["标的", "货物", "商品", "服务内容", "工程"],
    ClauseType.PRICE: ["价款", "金额", "费用", "报酬", "价格", "元"],
    ClauseType.PERFORMANCE: ["交付", "履行", "期限", "地点", "方式"],
    ClauseType.QUALITY: ["质量", "标准", "规格", "验收", "合格"],
    ClauseType.BREACH: ["违约", "赔偿", "违约金", "罚金", "滞纳金"],
    ClauseType.DISPUTE: ["争议", "仲裁", "诉讼", "调解", "协商"],
    ClauseType.JURISDICTION: ["管辖", "法院", "仲裁委"],
    ClauseType.CONFIDENTIAL: ["保密", "商业秘密", "不得泄露"],
    ClauseType.FORCE_MAJEURE: ["不可抗力", "自然灾害", "疫情", "战争"],
    ClauseType.NOTICE: ["通知", "送达", "告知", "书面"],
    ClauseType.AMENDMENT: ["变更", "修改", "补充协议"],
    ClauseType.TERMINATION: ["终止", "解除", "到期", "续期"],
}

def classify_clause(text: str) -> ClauseType:
    scores = {ct: sum(1 for kw in kws if kw in text) for ct, kws in CLAUSE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ClauseType.OTHER

# ── 规则引擎 (模拟400+规则) ──
RISK_RULES = [
    {"id": "R001", "name": "违约金过高", "pattern": r'违约金.*?(\d+)%', "check": lambda m: int(m.group(1)) > 30,
     "risk": RiskLevel.HIGH, "desc": "违约金超过合同金额30%可能被法院调减",
     "statute": "《民法典》第585条", "suggestions": ["保守: 将违约金降至合同金额的10%", "平衡: 将违约金调整为20%并约定损失证明义务", "激进: 保留当前比例但增加实际损失条款"]},
    {"id": "R002", "name": "管辖约定模糊", "pattern": r'向.{0,10}法院', "check": lambda m: "人民法院" not in m.group(),
     "risk": RiskLevel.MEDIUM, "desc": "管辖法院约定不够具体，可能导致管辖权争议",
     "statute": "《民事诉讼法》第35条", "suggestions": ["明确约定'向XX市XX区人民法院提起诉讼'"]},
    {"id": "R003", "name": "缺少不可抗力条款", "pattern": None, "check": lambda text: "不可抗力" not in text,
     "risk": RiskLevel.MEDIUM, "desc": "合同未约定不可抗力条款，疫情等场景下风险敞口大",
     "statute": "《民法典》第180条", "suggestions": ["建议增加不可抗力条款，明确通知义务和责任免除范围"]},
    {"id": "R004", "name": "付款条件缺失", "pattern": None, "check": lambda text: "付款" not in text and "支付" not in text,
     "risk": RiskLevel.HIGH, "desc": "合同未明确付款条件、时间和方式",
     "statute": "《民法典》第511条", "suggestions": ["明确约定付款时间节点、支付方式和逾期利息"]},
    {"id": "R005", "name": "保密期限过长", "pattern": r'保密期限.*?(\d+)年', "check": lambda m: int(m.group(1)) > 5,
     "risk": RiskLevel.LOW, "desc": "保密期限超过5年，可能被认定为不合理限制",
     "statute": "《反不正当竞争法》第9条", "suggestions": ["建议将保密期限调整为3年，或约定'至信息公开之日'"]},
    {"id": "R006", "name": "争议解决方式缺失", "pattern": None, "check": lambda text: "仲裁" not in text and "诉讼" not in text and "法院" not in text,
     "risk": RiskLevel.MEDIUM, "desc": "合同未约定争议解决方式",
     "statute": "《仲裁法》第4条", "suggestions": ["建议约定仲裁或诉讼管辖，避免争议时管辖不明"]},
]

class RuleEngine:
    """合同风险规则引擎 (模拟400+规则)"""
    def scan(self, full_text: str, clauses: list[Clause]) -> list[dict]:
        findings = []
        for rule in RISK_RULES:
            if rule["pattern"]:
                m = re.search(rule["pattern"], full_text)
                if m and rule["check"](m):
                    findings.append({"rule_id": rule["id"], "name": rule["name"], "risk": rule["risk"],
                        "desc": rule["desc"], "statute": rule["statute"], "suggestions": rule["suggestions"],
                        "matched_text": m.group()})
            else:
                if rule["check"](full_text):
                    findings.append({"rule_id": rule["id"], "name": rule["name"], "risk": rule["risk"],
                        "desc": rule["desc"], "statute": rule["statute"], "suggestions": rule["suggestions"],
                        "matched_text": ""})
        return findings

# ── 合同审查主服务 ──
class ContractReviewer:
    """合同审查服务 — UC9"""
    def __init__(self):
        self.rule_engine = RuleEngine()

    def parse_contract(self, raw_text: str) -> list[Clause]:
        """将合同文本切分为条款"""
        sections = re.split(r'(第[一二三四五六七八九十百]+条|第\d+条)', raw_text)
        clauses = []
        for i in range(1, len(sections), 2):
            header = sections[i]
            body = sections[i+1].strip() if i+1 < len(sections) else ""
            full = header + " " + body
            ct = classify_clause(full)
            clauses.append(Clause(f"CL_{i//2+1:03d}", ct, full, i//2))
        if not clauses and raw_text.strip():
            clauses.append(Clause("CL_001", classify_clause(raw_text), raw_text, 0))
        return clauses

    def review(self, raw_text: str, contract_type: str = "买卖合同", perspective: str = "我方") -> ReviewReport:
        """执行合同审查"""
        # Step 1: 条款切分与分类
        clauses = self.parse_contract(raw_text)
        # Step 2: 规则引擎扫描
        findings = self.rule_engine.scan(raw_text, clauses)
        # Step 3: 将风险关联到条款
        for f in findings:
            for c in clauses:
                if f["matched_text"] and f["matched_text"] in c.text:
                    c.risks.append(f); c.suggestions.extend(f["suggestions"])
                    if f["risk"].value < c.risk_level.value or c.risk_level == RiskLevel.NONE:
                        c.risk_level = f["risk"]
                    break
            else:
                if clauses: clauses[-1].risks.append(f); clauses[-1].suggestions.extend(f["suggestions"])
        # Step 4: 合规度评分
        high = sum(1 for f in findings if f["risk"] == RiskLevel.HIGH)
        med = sum(1 for f in findings if f["risk"] == RiskLevel.MEDIUM)
        low = sum(1 for f in findings if f["risk"] == RiskLevel.LOW)
        score = max(0, 100 - high * 20 - med * 10 - low * 5)
        return ReviewReport(contract_type, perspective, len(clauses), high, med, low, score, clauses)

def demo():
    print("=" * 60); print("  合同审查模块 — 可运行 Demo"); print("=" * 60)
    reviewer = ContractReviewer()
    contract = """
第一条 甲方（出卖人）：某科技有限公司，乙方（买受人）：某贸易有限公司。
第二条 标的物为A型号设备100台，单价5万元，合同总金额为500万元。
第三条 甲方应于2024年6月30日前将货物交付至乙方指定地点。
第四条 货物应符合国家标准GB/T-XXXX，乙方收货后7日内完成验收。
第五条 如甲方逾期交付，应按合同金额的50%支付违约金。
第六条 保密期限为10年，双方不得泄露合同内容及商业秘密。
第七条 争议由双方协商解决。
"""
    print(f"\n📄 待审查合同:\n{contract}")
    report = reviewer.review(contract, "买卖合同", "买方")
    print(f"\n{'='*60}")
    print(f"📋 审查报告")
    print(f"{'='*60}")
    print(f"  合同类型: {report.contract_type} | 审查角度: {report.perspective}")
    print(f"  条款总数: {report.total_clauses}")
    print(f"  🔴 高风险: {report.high_risk} | 🟡 中风险: {report.medium_risk} | 🟢 低风险: {report.low_risk}")
    print(f"  📊 合规度评分: {report.compliance_score}/100")
    print(f"\n  条款详情:")
    for c in report.clauses:
        icon = {"高风险":"🔴","中风险":"🟡","低风险":"🟢","无风险":"⚪"}.get(c.risk_level.value, "?")
        print(f"    {icon} [{c.clause_type.value}] {c.text[:50]}...")
        for r in c.risks:
            print(f"       ⚠️  {r['name']}: {r['desc']}")
            print(f"       📖 依据: {r['statute']}")
            for s in r["suggestions"]: print(f"       💡 {s}")

if __name__ == "__main__":
    demo()
