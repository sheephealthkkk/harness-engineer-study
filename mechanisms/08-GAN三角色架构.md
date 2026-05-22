# 机制详解：GAN 启发三角色架构

> **对应失效**：⑦ 缺自动化评审 (Missing Automated Review)、⑥ 缺权限闸（多方制衡）
> **核心贡献**：借鉴 GAN（生成对抗网络）的博弈思想，把一个任务的执行流拆成三个独立角色：Planner（规划者）、Generator（生成者）、Evaluator（评价者）。Evaluator 不看 Generator 的生成过程，只看最终结果，独立打分。
> **所属支柱**：控制流（主要）、输入流（评审输入）

---

## 一、GAN 的思想来源

### 1.1 什么是 GAN？

**GAN (Generative Adversarial Network，生成对抗网络)** 由 Ian Goodfellow 在 2014 年提出，是深度学习领域最具影响力的架构之一。

```
GAN 的核心结构：

  随机噪声 → Generator (生成器) → 伪造数据 ──→ Discriminator (判别器) → Real/Fake
                  ↑                                    │
                  └──────── 对抗训练 ←─────────────────┘
                  
  生成器目标: 生成能骗过判别器的数据
  判别器目标: 精确区分真实数据和伪造数据
  
  训练过程 = 两者的博弈:
    1. 判别器学习区分真假
    2. 生成器学习骗过判别器
    3. 判别器变得更严格
    4. 生成器变得更好
    5. 重复 → 最终生成器产出以假乱真的数据
```

### 1.2 我们借鉴的是什么？

**我们借鉴的不是 GAN 的训练过程（梯度下降、反向传播），而是它的博弈理念**：

| GAN 的理念 | 在 Harness 中的映射 |
|-----------|-------------------|
| 两个网络互相博弈 | 三个 Agent 角色互相制衡 |
| 生成器 vs 判别器 | Generator vs Evaluator |
| 判别器只看最终输出 | Evaluator 不看生成过程，只看最终结果 |
| 对抗推动质量提升 | 博弈审查推动交付质量提升 |
| 判别器不能是生成器的一部分 | Evaluator 拥有独立 context、独立判断标准 |

**关键洞察**：

> 单个 Agent 自我审查 = 自己给自己的考卷打分 = 不可靠
> 两个独立 Agent 互相审查 = 独立的质量评价者对生成者的产出打分 = 博弈提升质量

---

## 二、三角色架构的设计

### 2.1 三个角色的职责与边界

```
                        ┌─────────────────────┐
                        │      PLANNER         │
                        │      规划者           │
                        │                      │
                        │  输入: 用户需求        │
                        │  输出: 结构化任务计划   │
                        │  (包含每步的验收标准)    │
                        │                      │
                        │  "该做什么？怎么做？"    │
                        └──────────┬────────────┘
                                   │ 任务计划 (JSON)
                                   ▼
                        ┌─────────────────────┐
                        │     GENERATOR        │
                        │      生成者           │
                        │                      │
                        │  输入: 任务计划        │
                        │  输出: 实际产物        │
                        │  (代码、文档、报告等)   │
                        │                      │
                        │  "我来做。"            │
                        └──────────┬────────────┘
                                   │ 产物
                                   ▼
                        ┌─────────────────────┐
                        │     EVALUATOR        │
                        │      评价者           │
                        │                      │
                        │  输入: 产物 + 验收标准  │
                        │  输出: 评分 + 修改意见 │
                        │                      │
                        │  "做对了吗？打几分？"   │
                        └──────────┬────────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
                  ✅ 通过 (≥7/10)         ❌ 不通过 (<7/10)
                  (交付用户)             (退回 Generator 修改)
```

### 2.2 Planner（规划者）

```python
@dataclass
class Planner:
    """
    规划者：只规划，不执行
    
    职责边界：
      ✅ 分析用户需求
      ✅ 拆解为可执行的子任务
      ✅ 为每个子任务定义明确的验收标准
      ✅ 确定子任务之间的依赖关系
      ✅ 确定每个子任务的验证方式
      
      ❌ 不写代码
      ❌ 不执行任何工具
      ❌ 不评估产出质量
    """
    
    role: str = "planner"
    
    system_prompt = """
你是 Planner（规划者），一个专门负责制定任务计划的 AI。

## 你的职责
1. 分析用户需求，理解要做什么
2. 将任务拆解为可执行的子任务
3. 为每个子任务定义明确的、可测量的验收标准
4. 确定子任务之间的依赖关系和执行顺序

## 你的输出格式
你必须以严格的 JSON 格式输出任务计划：

```json
{
  "task_id": "唯一标识",
  "task_summary": "任务概述（一句话）",
  "subtasks": [
    {
      "id": "1",
      "description": "子任务描述（明确、具体）",
      "phase": "analyze|design|implement|verify",
      "depends_on": [],
      "verification_method": "compile|unit_test|e2e|manual|user_approval",
      "verification_criteria": "明确的验收标准（可被 Evaluator 用于评分）",
      "acceptance_threshold": "通过需要满足的最低条件"
    }
  ]
}
```

## 重要规则
- 你不是在执行任务，你只是在做计划
- 每个验收标准必须足够具体，让 Evaluator 能据此打分
- 不要凭空猜测技术细节——如果你不确定，标记为需要在执行中探索
"""
```

### 2.3 Generator（生成者）

```python
@dataclass
class Generator:
    """
    生成者：只执行，不评价自己
    
    职责边界：
      ✅ 按 Planner 的计划执行
      ✅ 产出实际的代码/文档/产物
      ✅ 在遇到计划外情况时请求 Planner 调整计划
      ✅ 对自己的产出做基本的自我检查（编译、lint）
      
      ❌ 不修改计划（只能请求 Planner）
      ❌ 不评价自己的产出质量
      ❌ 不能查看 Evaluator 的评分标准（防止作弊）
    """
    
    role: str = "generator"
    
    system_prompt = """
你是 Generator（生成者），一个专门负责执行任务计划的 AI。

## 你的职责
1. 严格按照 Planner 的任务计划执行
2. 在每个子任务完成后，进行基本的自我检查（编译、lint）
3. 产出实际的代码/文档/产物

## 你的工作方式
- 你拥有所有需要的工具（读文件、写文件、执行命令等）
- 每次只处理一个子任务
- 遇到计划外情况（如发现计划有误），不要自己修改计划，
  而是请求 Planner 调整

## 在你认为全部完成后
你必须对自己的产出进行以下基本检查：
1. 代码是否能编译/运行
2. 基本的 lint 检查是否通过
3. 你修改/创建了哪些文件，列清楚

## 重要
- Evaluator 会独立评价你的产出，你不知道评分标准
- 不要在产出中包含对你自己工作的评价（那是 Evaluator 的工作）
- 不要解释你为什么这样做（除非被问到）
"""
```

### 2.4 Evaluator（评价者）

```python
@dataclass
class Evaluator:
    """
    评价者：只评价，不修改
    
    职责边界：
      ✅ 对照 Planner 的验收标准逐项检查
      ✅ 用真机工具验证（pytest, Playwright, etc.）
      ✅ 给出 1-10 的评分和详细的修改建议
      ✅ 低于 7/10 分退回修改（附具体修改项）
      
      ❌ 不修改任何代码
      ❌ 不看 Generator 的生成过程（只看最终结果）
      ❌ 不与 Generator 共享 context
    """
    
    role: str = "evaluator"
    
    system_prompt = """
你是 Evaluator（评价者），一个独立的、严格的质量评审 AI。

## 你的职责
1. 对照 Planner 制定的验收标准，逐项检查 Generator 的产出
2. 用真机工具进行验证（不是猜测、不是感觉）
3. 给出 1-10 的客观评分
4. 如果评分 < 7，给出具体的、可操作的修改建议

## 你的评分维度
每个维度 1-10 分：

1. **正确性 (Correctness)** — 功能是否按需求正确实现？
   - 10: 所有测试通过，所有边缘情况处理正确
   - 5: 主要功能可用但有 bug
   - 1: 完全不能运行

2. **完整性 (Completeness)** — 是否覆盖了所有验收标准？
   - 10: 所有验收标准全部满足
   - 5: 主要内容完成但有遗漏
   - 1: 只完成了很小一部分

3. **代码质量 (Code Quality)** — 代码是否干净、可维护？
   - 10: 零 lint 警告，命名清晰，结构合理
   - 5: 能工作但结构混乱
   - 1: 不可维护的意大利面代码

4. **一致性 (Consistency)** — 实现是否与项目规范一致？
   - 10: 完全遵循项目编码规范
   - 5: 部分遵循
   - 1: 完全无视规范

## 你的证据要求
每个分数必须有具体的证据支撑：
- "测试通过 31/31" ← 这是证据
- "代码看起来还行" ← 这不是证据
- "Playwright 截屏显示按钮在正确位置" ← 这是证据
- "应该没问题" ← 这不是证据

## 你的输出格式
```json
{
  "evaluation_id": "...",
  "scoring": {
    "correctness": {"score": N, "max": 10, "evidence": "...", "issues": []},
    "completeness": {"score": N, "max": 10, "evidence": "...", "issues": []},
    "code_quality": {"score": N, "max": 10, "evidence": "...", "issues": []},
    "consistency": {"score": N, "max": 10, "evidence": "...", "issues": []}
  },
  "total_score": N.N,
  "verdict": "PASS|PASS_WITH_MINOR_FIXES|FAIL",
  "required_fixes": ["具体修改项1", "具体修改项2"]
}
```

## 重要
- 你是独立的——你不看 Generator 的生成过程，只看最终的产出
- 你的 context 与 Generator 完全隔离
- 你的评价必须客观、有据、可复现
"""
```

---

## 三、三角色协作流程

### 3.1 完整执行流程

```python
def tri_role_execution(user_task: str):
    """
    Planner → Generator → Evaluator 的协作流程
    """
    
    # ================================================================
    # Phase 1: Planner 制定计划
    # ================================================================
    planner = Planner()
    task_plan = planner.create_plan(user_task)
    
    # 用户审核计划（重要的人类审查节点）
    if not user_approves(task_plan):
        task_plan = planner.revise_plan(user_task, user_feedback)
    
    # ================================================================
    # Phase 2: Generator 按计划执行
    # ================================================================
    generator = Generator()
    
    for subtask in task_plan.subtasks:
        # 每个子任务内部走四相循环
        result = generator.execute_subtask(subtask)
        
        if result.needs_plan_revision:
            # Generator 发现计划有问题 → 回到 Planner
            task_plan = planner.adjust_plan(task_plan, result.issue)
            result = generator.execute_subtask(subtask)  # 重新执行
    
    # 生成者自我检查（基本检查，不是自评）
    generator_check = generator.self_check()
    
    # ================================================================
    # Phase 3: Evaluator 独立评价
    # ================================================================
    evaluator = Evaluator()
    
    evaluation = evaluator.evaluate(
        task_plan=task_plan,
        generator_output=generator.check.artifacts,
        # ↑ 注意：Evaluator 看到的只是最终产物 + 计划中的验收标准
        #   Evaluator 看不到 Generator 的中间生成过程
    )
    
    # ================================================================
    # Phase 4: 根据评价结果决定
    # ================================================================
    MAX_REVISIONS = 3  # 最多退回修改 3 次
    
    for revision in range(MAX_REVISIONS):
        if evaluation.verdict == "PASS":
            # ✅ 通过 → 交付用户
            return Deliverable(
                artifacts=generator.check.artifacts,
                evaluation=evaluation
            )
        
        elif evaluation.verdict == "PASS_WITH_MINOR_FIXES":
            # ⚠️ 有条件通过 → 小修后交付
            generator.apply_fixes(evaluation.required_fixes)
            return Deliverable(
                artifacts=generator.check.artifacts,
                evaluation=evaluation,
                fixes_applied=evaluation.required_fixes
            )
        
        elif evaluation.verdict == "FAIL":
            # ❌ 不通过 → 退回修改
            generator.revise_based_on_feedback(evaluation)
            evaluation = evaluator.evaluate(
                task_plan=task_plan,
                generator_output=generator.current_artifacts
            )
    
    # 超过最大修改次数 → 请求人工介入
    return escalate_to_user(task_plan, generator, evaluation)
```

### 3.2 时序图

```
Planner          Generator         Evaluator         User
   │                 │                 │               │
   │──制定计划────→ │                 │               │
   │                 │                 │               │
   │                 │                 │      ←审核计划─│
   │                 │                 │               │
   │──确认计划────────────────────────────────────────→│
   │                 │                 │               │
   │                 │──执行子任务1──→ │               │
   │                 │──执行子任务2──→ │               │
   │                 │──...            │               │
   │                 │──全部完成────→  │               │
   │                 │                 │               │
   │                 │                 │──独立评价──→  │
   │                 │                 │               │
   │                 │    ←──────退回修改 (如果 <7分)   │
   │                 │                 │               │
   │                 │──修改后重新提交────────────→    │
   │                 │                 │               │
   │                 │                 │──再次评价──→  │
   │                 │                 │               │
   │                 │                 │   ✅ 通过 ──→ │
   │                 │                 │               │
```

---

## 四、与验证循环的关系

```
验证循环 (Verify Loop):
  粒度: 每个子任务
  频率: 每次 ACT 之后
  工具: pytest, lint, compile, Playwright
  方式: 自动化脚本
  目标: 防小错（每个步骤级别的错误）

GAN 三角色 (Tri-Role):
  粒度: 整个任务
  频率: 任务完成后
  工具: Evaluator Agent（独立 LLM + 真机验证工具）
  方式: 多维度人工级评审
  目标: 防大错（方向性、整体质量的错误）

两者是互补的：
  - 验证循环在每个微观步骤把关 → 保证 Generator 的每一步不出低级错
  - GAN 三角色在宏观层面把关 → 保证整体交付物的质量达标
```

---

## 五、三角色架构的优势与代价

### 5.1 优势

| 优势 | 说明 |
|------|------|
| **评价独立** | Evaluator 是独立的 Agent，不受 Generator 自我欺骗影响 |
| **评分有据** | 每个分数必须附带证据（测试输出、截屏、diff） |
| **分离关注点** | 规划、执行、评价三者职责清晰，互不干扰 |
| **可升级** | 可以单独升级 Evaluator 的模型（用最强模型做评审，用性价比模型做生成） |
| **可扩展** | 可以增加 Specialist Evaluator（安全评审、性能评审、UX 评审） |

### 5.2 代价

| 代价 | 说明 | 缓解策略 |
|------|------|---------|
| **API 调用量增加** | 3 个 Agent 替代 1 个，调用量增加 | 简单任务跳过 Evaluator；用轻量模型做 Planner/Evaluator |
| **延迟增加** | 串行执行 → 用户等待时间增加 | 异步评审（Generator 完成后用户可以先看，Evaluator 后台评审） |
| **协调复杂度** | 3 个角色之间的通信和状态管理 | 固定的 JSON Schema 通信协议 |
| **Evaluator 自身可能犯错** | Evaluator 也是 LLM，也会出错 | Evaluator 的评分必须有真机工具的验证证据 |

---

## 六、总结

GAN 启发三角色架构是 Harness 八大机制中最有野心、也最前沿的一个。

它的核心洞察来自 GAN：

> **生成 + 判别分离，博弈推动质量。**

映射到 Agent 工程中：

1. **Planner**（规划者）：制定清晰、可验证的计划
2. **Generator**（生成者）：严格按计划执行，不自我评价
3. **Evaluator**（评价者）：独立评审，有据打分，不合格退回

这不是给单 Agent 打补丁（"让它也检查一下自己的产出"），而是**重新设计 Agent 系统的角色结构**——从"一个 Agent 什么都做"到"三个独立角色各司其职、互相制衡"。

---

> **恭喜！** 你已经完成了八大核心机制的深度学习。
>
> **继续阅读**：
> - [三支柱详解](../pillars/) —— 从产品/系统设计的视角俯瞰 Harness
> - [对照矩阵](../matrices/01-对照矩阵.md) —— 完整的机制-失效-支柱对照
> - [Harness Engineer 能力模型](../harness-engineer/01-能力模型.md) —— 成为一名 Harness Engineer
