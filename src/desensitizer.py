# -*- coding: utf-8 -*-
"""
隐私脱敏处理模块 (Privacy Desensitization Module)
=================================================
实现: 多级脱敏(L0-L3) + NER模拟 + 假名映射 + 审计日志
"""
from __future__ import annotations
import enum, hashlib, json, re, time, uuid
from dataclasses import dataclass, field
from typing import Optional

class DesensitizeLevel(enum.Enum):
    L0_NONE = 0; L1_PSEUDONYM = 1; L2_GENERALIZE = 2; L3_REDACT = 3

class EntityType(enum.Enum):
    PERSON="当事人"; COMPANY="法人/公司"; ID_NUMBER="身份证号"
    BANK_ACCOUNT="银行账号"; PHONE="手机号"; AMOUNT="金额"
    DATE="日期"; ADDRESS="地址"; CASE_NUMBER="案号"; STATUTE="法条引用"

@dataclass
class LegalEntity:
    entity_id: str; entity_type: EntityType; original_text: str
    start_pos: int; end_pos: int
    anonymized_text: str = ""; level: DesensitizeLevel = DesensitizeLevel.L1_PSEUDONYM

@dataclass
class SanitizeResult:
    sanitized_text: str; mapping_id: str
    entities_found: int; entities_sanitized: int; audit_entry: dict

class LegalNERSimulator:
    """法律NER模拟器 (生产中用Legal-BERT, F1≈0.93)"""
    PATTERNS = {
        EntityType.ID_NUMBER: re.compile(r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'),
        EntityType.PHONE: re.compile(r'\b1[3-9]\d{9}\b'),
        EntityType.AMOUNT: re.compile(r'[¥￥]\s*[\d,]+(?:\.\d{1,2})?万?元?|\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:万元|元|万)'),
        EntityType.DATE: re.compile(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?'),
        EntityType.CASE_NUMBER: re.compile(r'[\(（]\d{4}[\)）]\s*[\u4e00-\u9fa5]+\d+号'),
        EntityType.STATUTE: re.compile(r'《[\u4e00-\u9fa5]+》(?:第\d+条)?'),
    }
    PERSON_PAT = re.compile(r'(?:当事人|原告|被告|委托人|甲方|乙方)[：:]\s*([\u4e00-\u9fa5]{2,4})')
    COMPANY_PAT = re.compile(r'([\u4e00-\u9fa5]+(?:有限公司|股份公司|集团|事务所))')

    def recognize(self, text: str) -> list[LegalEntity]:
        entities = []
        for et, pat in self.PATTERNS.items():
            for m in pat.finditer(text):
                entities.append(LegalEntity(f"E_{uuid.uuid4().hex[:8]}", et, m.group(), m.start(), m.end()))
        for m in self.PERSON_PAT.finditer(text):
            entities.append(LegalEntity(f"E_{uuid.uuid4().hex[:8]}", EntityType.PERSON, m.group(1), m.start(1), m.end(1)))
        for m in self.COMPANY_PAT.finditer(text):
            entities.append(LegalEntity(f"E_{uuid.uuid4().hex[:8]}", EntityType.COMPANY, m.group(1), m.start(1), m.end(1)))
        entities.sort(key=lambda e: e.start_pos)
        return entities

DEFAULT_LEVEL_POLICY = {
    EntityType.STATUTE: DesensitizeLevel.L0_NONE, EntityType.CASE_NUMBER: DesensitizeLevel.L0_NONE,
    EntityType.PERSON: DesensitizeLevel.L1_PSEUDONYM, EntityType.COMPANY: DesensitizeLevel.L1_PSEUDONYM,
    EntityType.AMOUNT: DesensitizeLevel.L2_GENERALIZE, EntityType.DATE: DesensitizeLevel.L2_GENERALIZE,
    EntityType.ADDRESS: DesensitizeLevel.L2_GENERALIZE,
    EntityType.ID_NUMBER: DesensitizeLevel.L3_REDACT, EntityType.BANK_ACCOUNT: DesensitizeLevel.L3_REDACT,
    EntityType.PHONE: DesensitizeLevel.L3_REDACT,
}

class Desensitizer:
    """多级脱敏处理器 (L0不脱敏/L1假名化/L2泛化/L3完全删除)"""
    def __init__(self, level_policy=None):
        self.ner = LegalNERSimulator()
        self.level_policy = level_policy or DEFAULT_LEVEL_POLICY
        self.mapping_store: dict[str, dict] = {}
        self.audit_log: list[dict] = []
        self._counters: dict[str, int] = {}

    def _next_pseudo(self, et: EntityType) -> str:
        k = et.name; c = self._counters.get(k, 0); self._counters[k] = c + 1
        s = chr(ord('A') + c) if c < 26 else str(c)
        m = {EntityType.PERSON: f"[PARTY_{s}]", EntityType.COMPANY: f"[COMPANY_{s}]"}
        return m.get(et, f"[{k}_{s}]")

    def _generalize(self, e: LegalEntity) -> str:
        if e.entity_type == EntityType.AMOUNT:
            nums = re.findall(r'[\d,.]+', e.original_text)
            if nums:
                try:
                    v = float(nums[0].replace(',',''))
                    if '万' in e.original_text: v *= 10000
                    return "[AMOUNT:大额]" if v >= 1e6 else "[AMOUNT:中额]" if v >= 1e5 else "[AMOUNT:小额]"
                except: pass
            return "[AMOUNT:未知]"
        elif e.entity_type == EntityType.DATE: return "[DATE:近期]"
        return f"[{e.entity_type.name}:泛化]"

    def sanitize(self, text: str) -> SanitizeResult:
        self._counters.clear()
        entities = self.ner.recognize(text)
        mapping, result_text, cnt = {}, text, 0
        for e in reversed(entities):
            lv = self.level_policy.get(e.entity_type, DesensitizeLevel.L1_PSEUDONYM); e.level = lv
            if lv == DesensitizeLevel.L0_NONE: continue
            elif lv == DesensitizeLevel.L1_PSEUDONYM: t = self._next_pseudo(e.entity_type); mapping[t] = e.original_text
            elif lv == DesensitizeLevel.L2_GENERALIZE: t = self._generalize(e); mapping[t] = e.original_text
            else: t = "[REDACTED]"
            e.anonymized_text = t
            result_text = result_text[:e.start_pos] + t + result_text[e.end_pos:]
            cnt += 1
        mid = f"MAP_{uuid.uuid4().hex[:12]}"; self.mapping_store[mid] = mapping
        audit = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mapping_id": mid,
                 "entities_found": len(entities), "entities_sanitized": cnt,
                 "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16]}
        self.audit_log.append(audit)
        return SanitizeResult(result_text, mid, len(entities), cnt, audit)

    def restore(self, sanitized_text: str, mapping_id: str) -> str:
        mapping = self.mapping_store.get(mapping_id, {})
        if not mapping: raise ValueError(f"映射表 {mapping_id} 不存在")
        r = sanitized_text
        for tok, orig in mapping.items(): r = r.replace(tok, orig)
        return r

    def get_audit_report(self) -> list[dict]: return self.audit_log

def demo():
    print("=" * 60); print("  隐私脱敏处理模块 — 可运行 Demo"); print("=" * 60)
    d = Desensitizer()
    text = ("原告：张三，身份证号110101199001011234，手机号13800138000。"
            "被告：北京某科技有限公司。案情：原告于2024-03-15与被告签订买卖合同，"
            "合同金额为¥500万元。原告依据《民法典》第577条提起诉讼。案号为（2024）京民终123号。")
    print(f"\n📄 原始文本:\n   {text}")
    r = d.sanitize(text)
    print(f"\n🔒 脱敏后:\n   {r.sanitized_text}")
    print(f"\n📊 统计: 发现{r.entities_found}个实体, 脱敏{r.entities_sanitized}个, 映射ID={r.mapping_id}")
    restored = d.restore(r.sanitized_text, r.mapping_id)
    print(f"\n🔓 还原:\n   {restored}")
    print(f"\n📋 审计: {json.dumps(r.audit_entry, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    demo()
