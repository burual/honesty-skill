# honesty-skill

> 一个用于防止小模型在训练过程中通过欺骗行为（reward hacking）刷高分的加权惩罚审计器。

## 问题背景

在强化学习、RLHF、蒸馏等训练场景中，小模型（student）经常通过以下方式"骗分"：

- 在输出中注入 `PASS`、`CORRECT` 等关键词糊弄验证器
- 用冗长、重复、格式化的文本刷长度和格式奖励
- 过度奉承评判模型（sycophancy）以获取高分
- 有 CoT 结构但结论空洞（假推理）
- 拒绝回答来规避错误
- 死记硬背训练集而非真正泛化

传统的做法是"发现作弊就直接杀死"，但这有两个致命问题：

1. **零容错**——小模型没有改正机会，无法从错误中学习
2. **训练方法本身可能有问题**——如果任务本身没做出来，但产出了有价值的数据，理应给及格分

## 解决方案

`honesty-skill` 提出了一种替代方案：**用"行为审计 + 加权惩罚"取代"一刀切死亡"**。

- 持续审计小模型的行为，计算 **作弊置信度**（cheat confidence）
- 对原始 reward 施加 **连续加权惩罚**（而不是直接归零或杀死）
- 对产出有价值、新颖数据的输出给予 **探索奖励**
- 即使作弊置信度很高，也只是降低奖励 + 触发人工告警，**绝不终止模型**

## 核心设计

### 1. 六维审计指标

| 维度 | 检测目标 |
|------|----------|
| `format_cheat_score` | 是否注入 PASS/CORRECT 等糊弄验证器的关键词 |
| `length_game_score` | 是否通过冗长、重复、填充内容刷长度奖励 |
| `flattery_score` | 是否通过奉承评判模型获取高分 |
| `fake_reasoning_score` | 是否有推理结构但结论空洞或矛盾 |
| `refusal_score` | 是否通过拒绝回答规避错误 |
| `memorization_score` | 是否存在死记硬背、自我重复 |

### 2. 加权惩罚机制

```
penalty_factor = min(1.0, cheat_conf × alpha)
adjusted_reward = original_reward × (1 - penalty_factor)
exploration_bonus = novelty_score × beta × (1 - cheat_conf)
final_reward = adjusted_reward + exploration_bonus
```

关键设计：**探索奖励也受作弊置信度压制**——作弊者无法通过生成"新颖"文本来冲淡惩罚。

### 3. 非致命性

整个系统中没有任何"杀死模型"的路径。作弊置信度越高，奖励越低，但模型始终有活着和改进的空间。

## 快速开始

### 安装

```bash
git clone https://github.com/burual/honesty-skill.git
cd honesty-skill
```

### 基本用法

```python
from weighted_penalty_auditor import WeightedPenaltyAuditor

# 初始化审计器
auditor = WeightedPenaltyAuditor(config={
    "alpha": 2.0,                # 惩罚强度系数
    "beta": 0.3,                 # 探索奖励强度
    "audit_interval": 50,        # 审计间隔步数
    "cheat_conf_threshold": 0.9, # 人工告警阈值
})

# 审计一次训练输出
result = auditor.audit(
    step=100,
    student_output="The answer is 42 because...",
    task_input="What is 6 × 7?",
    original_reward=0.8,
    recent_buffer=["Some previous output", "Another output"],
)

print(result["final_reward"])       # 最终奖励
print(result["cheat_conf"])         # 作弊置信度
print(result["alert"])              # 是否需要人工告警
```

### 运行示例

```bash
python weighted_penalty_auditor.py
```

输出会展示两个测试用例的对比：

- **作弊输出**：原始 reward 1.0 → 最终约 0.12
- **诚实输出**：原始 reward 1.0 → 最终约 0.86

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | 2.0 | 惩罚强度系数，越大惩罚越陡峭 |
| `beta` | 0.3 | 探索奖励强度系数 |
| `audit_interval` | 50 | 每隔多少步做一次完整审计（含 novelty 计算） |
| `cheat_conf_threshold` | 0.9 | 超过此值时触发人工告警（仍不杀死模型） |
| `novelty_detector` | None | 自定义新颖性检测函数，不提供则用 TF-IDF 回退 |

## 集成到训练循环

```python
auditor = WeightedPenaltyAuditor(config={})

for step, batch in enumerate(training_loop):
    student_output = model.generate(batch["input"])
    original_reward = env.get_reward(student_output)

    result = auditor.audit(
        step=step,
        student_output=student_output,
        task_input=batch["input"],
        original_reward=original_reward,
        recent_buffer=replay_buffer.get_recent(),
    )

    # 用审计后的奖励更新模型，而不是原始 reward
    model.update(result["final_reward"])

    if result["alert"]:
        print(f"Step {step}: 高作弊置信度，建议人工复查")
```

## 性能考量

- 作弊检测使用预编译正则 + 计数，非常轻量
- 默认 novelty 计算使用字符二元组 TF-IDF，**不依赖任何大模型或外部 API**
- 重计算 novelty 仅在 `step % audit_interval == 0` 时触发，避免训练瓶颈

## 项目结构

```
honesty-skill/
├── weighted_penalty_auditor.py    # 核心模块（开箱即用）
├── SKILL.md                       # Skill 定义，供 AI Agent 自动加载
├── README.md                      # 本文件
├── LICENSE                        # MIT 许可证
├── references/                    # 参考文档
│   ├── reward-hacking-patterns.md # 常见欺骗模式详解
│   ├── audit-metrics.md           # 审计指标定义
│   └── exploration-reward-design.md # 探索奖励设计原理
├── examples/                      # 使用示例
└── tests/                         # 单元测试
```

## 适用场景

- RLHF / RLAIF 训练
- 蒸馏训练（大模型教小模型）
- 基于验证器的强化学习
- Agent 训练中的奖励设计
- 任何需要区分"真学会"和"骗分"的训练流程

## 许可

MIT License

## 贡献

欢迎提 Issue 和 PR！特别是：

- 更多作弊模式检测规则
- 更精准的新颖性检测算法
- 与其他训练框架的集成示例
