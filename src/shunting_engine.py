# -*- coding: utf-8 -*-
"""
端云智能分流引擎 (Edge-Cloud Shunting Engine)
=============================================

实现要点：
  1. 数据分级 (D0~D3) — 硬规则优先路由
  2. 五维度加权决策矩阵 — 动态打分
  3. 三层优雅降级策略

本模块为可运行的 demo，模拟了完整的路由决策流程。
"""

from __future__ import annotations

import enum
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 1. 数据分级枚举
# ──────────────────────────────────────────────
class DataLevel(enum.Enum):
    """数据敏感等级 D0‑D3"""
    D0_PUBLIC = 0       # 法条编号、公开判例号
    D1_LOW_SENS = 1     # 合同类型、争议焦点
    D2_SENSITIVE = 2    # 当事人姓名、合同金额
    D3_HIGH_SENS = 3    # 身份证号、银行账号


class RouteTarget(enum.Enum):
    LOCAL_LLM = "LOCAL"
    CLOUD_LLM = "CLOUD"
    RULE_TEMPLATE = "RULE_TEMPLATE"   # 最后兜底


class DegradeLevel(enum.Enum):
    NONE = 0
    LEVEL1_CLOUD_TIMEOUT = 1
    LEVEL2_NETWORK_DOWN = 2
    LEVEL3_LOCAL_OVERLOAD = 3


class UserMode(enum.Enum):
    AUTO = "智能(默认)"
    LOCAL_PRIVACY = "本地隐私模式"
    CLOUD_PRECISION = "云端高精度模式"


# ──────────────────────────────────────────────
# 2. 请求与路由结果
# ──────────────────────────────────────────────
@dataclass
class LegalRequest:
    """用户提交的法律文书请求"""
    request_id: str
    content: str                              # 案情文本
    task_type: str = "contract_draft"         # contract_draft / lawsuit / legal_opinion
    user_id: str = "lawyer_001"
    user_mode: UserMode = UserMode.AUTO       # 用户全局模式


@dataclass
class RoutingDecision:
    """路由决策结果"""
    target: RouteTarget
    reason: str
    data_level: DataLevel
    local_score: float = 0.0
    cloud_score: float = 0.0
    degrade_level: DegradeLevel = DegradeLevel.NONE
    sanitized_text: Optional[str] = None      # 脱敏后文本 (云端路由时)
    mapping_id: Optional[str] = None          # 脱敏映射表ID


# ──────────────────────────────────────────────
# 3. 敏感词规则 (简化版 — 实际用 Legal-BERT NER)
# ──────────────────────────────────────────────
# D3 高敏关键词
D3_PATTERNS = ["身份证号", "身份证", "银行账号", "银行卡号", "未公开证据"]
# D2 敏感关键词
D2_PATTERNS = ["当事人", "合同金额", "住址", "手机号", "电话"]
# D1 业务低敏
D1_PATTERNS = ["合同类型", "争议焦点", "案由"]


def classify_data_level(text: str) -> DataLevel:
    """
    数据分级分类器 (简化版)
    实际生产中使用 Legal-BERT NER + 正则匹配组合
    """
    for pattern in D3_PATTERNS:
        if pattern in text:
            return DataLevel.D3_HIGH_SENS
    for pattern in D2_PATTERNS:
        if pattern in text:
            return DataLevel.D2_SENSITIVE
    for pattern in D1_PATTERNS:
        if pattern in text:
            return DataLevel.D1_LOW_SENS
    return DataLevel.D0_PUBLIC


# ──────────────────────────────────────────────
# 4. 五维度决策矩阵
# ──────────────────────────────────────────────
@dataclass
class DimensionScores:
    """五个评估维度的得分 (0‑100)"""
    sensitivity: float = 50.0     # 数据敏感度 (越高 → 越倾向本地)
    complexity: float = 50.0      # 任务复杂度 (越高 → 越倾向云端)
    network: float = 80.0         # 网络质量   (越高 → 越倾向云端)
    sla: float = 50.0             # 实时性SLA  (越高 → 越倾向本地)
    cost: float = 50.0            # 成本预算   (越高 → 越倾向云端)


# 维度权重
WEIGHTS = {
    "sensitivity": 0.30,
    "complexity":  0.25,
    "network":     0.20,
    "sla":         0.15,
    "cost":        0.10,
}


class DecisionMatrix:
    """五维度加权决策矩阵"""

    def __init__(self, weights: dict = None):
        self.weights = weights or WEIGHTS

    def evaluate_sensitivity(self, text: str, data_level: DataLevel) -> float:
        """评估数据敏感度 (0‑100, 越高越倾向本地)"""
        base_scores = {
            DataLevel.D0_PUBLIC: 10,
            DataLevel.D1_LOW_SENS: 40,
            DataLevel.D2_SENSITIVE: 75,
            DataLevel.D3_HIGH_SENS: 100,
        }
        return base_scores.get(data_level, 50)

    def evaluate_complexity(self, text: str, task_type: str) -> float:
        """评估任务复杂度 (0‑100, 越高越倾向云端)"""
        complexity_map = {
            "template_fill": 20,        # 模板填充 → 简单
            "contract_draft": 50,       # 合同起草 → 中等
            "lawsuit": 70,              # 起诉状   → 较复杂
            "legal_opinion": 85,        # 法律意见书 → 复杂
            "multi_law_analysis": 95,   # 多法条交叉 → 极复杂
        }
        base = complexity_map.get(task_type, 60)
        # 文本越长通常越复杂
        length_factor = min(len(text) / 2000, 1.0) * 15
        return min(base + length_factor, 100)

    def probe_network_quality(self) -> float:
        """
        探测网络质量 (0‑100, 越高越倾向云端)
        demo 中随机模拟，生产中使用 ICMP/HTTP 探针
        """
        rtt_ms = random.uniform(30, 600)
        if rtt_ms > 500:
            return 10  # 弱网
        elif rtt_ms > 200:
            return 50
        else:
            return 90  # 优质网络

    def check_sla(self, task_type: str) -> float:
        """检查实时性SLA (0‑100, 越高越倾向本地/低延迟)"""
        sla_map = {
            "interactive_edit": 90,     # 交互式编辑 → 需低延迟
            "template_fill": 80,
            "contract_draft": 50,
            "lawsuit": 40,
            "batch_generate": 10,       # 批量生成 → 可容忍高延迟
        }
        return sla_map.get(task_type, 50)

    def check_budget(self, user_id: str) -> float:
        """检查云端预算 (0‑100, 越高越倾向云端/有预算)"""
        # demo: 模拟预算充足度
        return random.uniform(40, 90)

    def compute(self, text: str, data_level: DataLevel, task_type: str,
                user_id: str) -> tuple[float, float, DimensionScores]:
        """
        计算本地/云端综合得分

        Returns:
            (local_score, cloud_score, dimension_scores)
        """
        scores = DimensionScores(
            sensitivity=self.evaluate_sensitivity(text, data_level),
            complexity=self.evaluate_complexity(text, task_type),
            network=self.probe_network_quality(),
            sla=self.check_sla(task_type),
            cost=self.check_budget(user_id),
        )

        w = self.weights

        local_score = (
            scores.sensitivity * w["sensitivity"]
            + scores.sla * w["sla"]
            + (100 - scores.network) * w["network"]
            + (100 - scores.cost) * w["cost"]
        )

        cloud_score = (
            scores.complexity * w["complexity"]
            + scores.network * w["network"]
            + scores.cost * w["cost"]
            + (100 - scores.sensitivity) * w["sensitivity"]
        )

        return local_score, cloud_score, scores


# ──────────────────────────────────────────────
# 5. 网络与云端状态模拟
# ──────────────────────────────────────────────
class CloudStatus:
    """模拟云端服务状态"""

    def __init__(self, available: bool = True, rtt_ms: float = 80):
        self.available = available
        self.rtt_ms = rtt_ms
        self._circuit_open = False    # 熔断器

    def is_available(self) -> bool:
        return self.available and not self._circuit_open

    def trip_circuit(self):
        """触发熔断"""
        self._circuit_open = True
        print("    ⚡ 熔断器已开启，后续请求将自动走本地")

    def half_open(self):
        """半开探测"""
        self._circuit_open = False


# ──────────────────────────────────────────────
# 6. 端云分流引擎 (核心)
# ──────────────────────────────────────────────
class ShuntingEngine:
    """
    端云智能分流引擎

    路由优先级：
      P0: D3高敏数据 → 强制本地
      P1: 网络断开/云端熔断/超预算 → 强制本地
      P2: 用户选择本地隐私模式 → 强制本地
      P3: 用户选择云端模式且脱敏通过 → 进入云端评分
      P4: 无硬规则命中 → 五维度评分
    """

    def __init__(self):
        self.matrix = DecisionMatrix()
        self.cloud_status = CloudStatus()
        self.routing_history: list[RoutingDecision] = []

    def route_request(self, request: LegalRequest) -> RoutingDecision:
        """
        核心路由决策方法

        实现策略硬规则 + 五维度加权评分 + 降级处理
        """
        print(f"\n{'='*60}")
        print(f"📋 处理请求: {request.request_id}")
        print(f"   任务类型: {request.task_type}")
        print(f"   用户模式: {request.user_mode.value}")
        print(f"   内容摘要: {request.content[:60]}...")

        # ── Step 1: 数据分级 ──
        data_level = classify_data_level(request.content)
        print(f"\n   📊 数据分级: {data_level.name} (等级 {data_level.value})")

        # ── Step 2: P0 硬规则 — D3高敏 → 强制本地 ──
        if data_level == DataLevel.D3_HIGH_SENS:
            decision = RoutingDecision(
                target=RouteTarget.LOCAL_LLM,
                reason="P0: 命中D3高敏数据，禁止原文出端",
                data_level=data_level,
            )
            self._log_decision(decision)
            return decision

        # ── Step 3: P1 硬规则 — 云端不可用 → 强制本地 ──
        if not self.cloud_status.is_available():
            decision = RoutingDecision(
                target=RouteTarget.LOCAL_LLM,
                reason="P1: 云端不可用(熔断/断网)，启用本地离线模式",
                data_level=data_level,
                degrade_level=DegradeLevel.LEVEL2_NETWORK_DOWN,
            )
            self._log_decision(decision)
            return decision

        # ── Step 4: P2 硬规则 — 用户选择本地 ──
        if request.user_mode == UserMode.LOCAL_PRIVACY:
            decision = RoutingDecision(
                target=RouteTarget.LOCAL_LLM,
                reason="P2: 用户主动选择本地隐私模式",
                data_level=data_level,
            )
            self._log_decision(decision)
            return decision

        # ── Step 5: P3 用户选择云端 → 需脱敏校验 ──
        if request.user_mode == UserMode.CLOUD_PRECISION:
            if data_level.value >= DataLevel.D2_SENSITIVE.value:
                print("   ⚠️  用户选择云端，但含敏感数据，需先脱敏")
                # 实际调用 Desensitizer，此处模拟
                sanitized = f"[SANITIZED] {request.content[:30]}..."
                decision = RoutingDecision(
                    target=RouteTarget.CLOUD_LLM,
                    reason="P3: 用户选择云端 + 脱敏通过",
                    data_level=data_level,
                    sanitized_text=sanitized,
                    mapping_id=f"MAP_{request.request_id}",
                )
            else:
                decision = RoutingDecision(
                    target=RouteTarget.CLOUD_LLM,
                    reason="P3: 用户选择云端(无需脱敏)",
                    data_level=data_level,
                )
            self._log_decision(decision)
            return decision

        # ── Step 6: P4 五维度评分 ──
        local_score, cloud_score, dim = self.matrix.compute(
            request.content, data_level, request.task_type, request.user_id
        )

        print(f"\n   🎯 五维度评分:")
        print(f"      数据敏感度: {dim.sensitivity:.1f}  (权重 30%)")
        print(f"      任务复杂度: {dim.complexity:.1f}  (权重 25%)")
        print(f"      网络质量:   {dim.network:.1f}  (权重 20%)")
        print(f"      实时性SLA:  {dim.sla:.1f}  (权重 15%)")
        print(f"      成本预算:   {dim.cost:.1f}  (权重 10%)")
        print(f"      ────────────────────────")
        print(f"      本地得分: {local_score:.2f}")
        print(f"      云端得分: {cloud_score:.2f}")

        if local_score >= cloud_score:
            target = RouteTarget.LOCAL_LLM
            reason = f"P4: 五维度评分 → 本地优先 (L={local_score:.1f} ≥ C={cloud_score:.1f})"
        else:
            target = RouteTarget.CLOUD_LLM
            reason = f"P4: 五维度评分 → 云端优先 (C={cloud_score:.1f} > L={local_score:.1f})"

        sanitized = None
        mapping_id = None
        if target == RouteTarget.CLOUD_LLM and data_level.value >= 1:
            sanitized = f"[SANITIZED] {request.content[:30]}..."
            mapping_id = f"MAP_{request.request_id}"

        decision = RoutingDecision(
            target=target,
            reason=reason,
            data_level=data_level,
            local_score=local_score,
            cloud_score=cloud_score,
            sanitized_text=sanitized,
            mapping_id=mapping_id,
        )
        self._log_decision(decision)
        return decision

    def emergency_fallback(self, request: LegalRequest) -> RoutingDecision:
        """
        紧急降级 — 当本地模型也负载过高时
        降级为规则模板 + 关键词检索
        """
        print(f"\n   🚨 Level 3 降级: 本地模型过载，使用规则模板兜底")
        return RoutingDecision(
            target=RouteTarget.RULE_TEMPLATE,
            reason="Level3: 本地模型过载，降级为规则模板",
            data_level=classify_data_level(request.content),
            degrade_level=DegradeLevel.LEVEL3_LOCAL_OVERLOAD,
        )

    def _log_decision(self, decision: RoutingDecision):
        """记录路由决策"""
        self.routing_history.append(decision)
        icon = "🏠" if decision.target == RouteTarget.LOCAL_LLM else "☁️"
        if decision.target == RouteTarget.RULE_TEMPLATE:
            icon = "📋"
        print(f"\n   {icon} 路由决策: {decision.target.value}")
        print(f"   📝 原因: {decision.reason}")
        if decision.degrade_level != DegradeLevel.NONE:
            print(f"   ⚠️  降级等级: {decision.degrade_level.name}")


# ──────────────────────────────────────────────
# 7. 演示运行
# ──────────────────────────────────────────────
def demo():
    """端云分流引擎演示"""
    print("=" * 60)
    print("  端云智能分流引擎 — 可运行 Demo")
    print("=" * 60)

    engine = ShuntingEngine()

    # 场景1: D3高敏数据 → 强制本地
    req1 = LegalRequest(
        request_id="REQ-001",
        content="请审查以下合同，当事人身份证号为110101199001011234，银行账号6222...",
        task_type="contract_draft",
    )
    engine.route_request(req1)

    # 场景2: 用户选择本地隐私模式
    req2 = LegalRequest(
        request_id="REQ-002",
        content="就本案争议焦点，需要分析《民法典》第1043条的适用情况，涉及合同金额500万元",
        task_type="legal_opinion",
        user_mode=UserMode.LOCAL_PRIVACY,
    )
    engine.route_request(req2)

    # 场景3: 公开数据 + 自动模式 → 五维度评分
    req3 = LegalRequest(
        request_id="REQ-003",
        content="请根据《民法典》第577条和《合同法》相关规定，起草一份标准买卖合同模板",
        task_type="template_fill",
    )
    engine.route_request(req3)

    # 场景4: 复杂推理 + 自动模式 → 大概率走云端
    req4 = LegalRequest(
        request_id="REQ-004",
        content="本案涉及知识产权侵权与合同违约竞合，需要综合分析《民法典》《著作权法》《反不正当竞争法》的交叉适用，争议焦点包括赔偿金额计算方式、连带责任认定",
        task_type="multi_law_analysis",
    )
    engine.route_request(req4)

    # 场景5: 云端不可用 → 自动降级
    print(f"\n{'='*60}")
    print("  模拟云端故障...")
    engine.cloud_status.available = False
    req5 = LegalRequest(
        request_id="REQ-005",
        content="请起草一份房屋租赁合同，租赁标的位于北京市朝阳区",
        task_type="contract_draft",
    )
    engine.route_request(req5)
    engine.cloud_status.available = True   # 恢复

    # 统计
    print(f"\n{'='*60}")
    print("  📊 路由统计")
    print(f"{'='*60}")
    local_count = sum(1 for d in engine.routing_history if d.target == RouteTarget.LOCAL_LLM)
    cloud_count = sum(1 for d in engine.routing_history if d.target == RouteTarget.CLOUD_LLM)
    tmpl_count = sum(1 for d in engine.routing_history if d.target == RouteTarget.RULE_TEMPLATE)
    total = len(engine.routing_history)
    print(f"  总请求数: {total}")
    print(f"  🏠 本地路由: {local_count} ({local_count/total*100:.0f}%)")
    print(f"  ☁️  云端路由: {cloud_count} ({cloud_count/total*100:.0f}%)")
    print(f"  📋 规则模板: {tmpl_count} ({tmpl_count/total*100:.0f}%)")


if __name__ == "__main__":
    demo()
