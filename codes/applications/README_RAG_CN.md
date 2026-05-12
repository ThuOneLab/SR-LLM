# RAG Library 构建与使用指南

本文件说明如何为 **SR-LLM 通用符号回归**（`general_symbolic_regression.py`）构建和维护 RAG（Retrieval-Augmented Generation）知识库。

---

## 1. RAG 知识库是什么？

在 SR-LLM 框架中，RAG 知识库是一个基于 **Chroma** 的向量数据库，用于存储“如何组合符号以构成新符号”的**专家经验**。在每一轮 Evolution 中，LLM 会：

1. 根据当前表现最好的符号，去 RAG 库中检索相似的历史知识；
2. 参考这些知识的组合方式，生成新的符号组合；
3. 对优秀的表达式进行**反思（Reflection）**，将新的经验写回 RAG 库。

因此，一个高质量的 RAG 知识库能够显著提升 LLM 生成有效符号的能力，加速符号回归的收敛。

---

## 2. 前置条件

- 已安装 SR-LLM 环境（`conda activate sr-llm`）
- 已配置 OpenAI API Key：
  ```bash
  export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- 若在国内使用，建议配置代理地址（默认 `172.22.0.1:7890`，可在代码中修改）

---

## 3. 初始化 RAG 知识库

RAG 知识库本质上是一个本地文件夹 + SQLite 文件。首次运行时，**自动创建**，无需手动建表。

```python
from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT

# 指定存储路径（相对项目根目录）
memory_path = "codes/ragLibrary/memory_my_task"

agent = RAG_AGENT(
    port="7890",
    address="172.22.0.1",
    memory_path=memory_path,
    fewshot_num=2,
    reflection_num=1,
    extend_num=1,
)
```

运行后，会在 `codes/ragLibrary/memory_my_task/` 下生成：
- `chroma.sqlite3` —— Chroma 向量数据库文件
- `knowledge_targets_names.json` —— 知识目标名称索引（自动维护）

---

## 4. 添加知识（Knowledge）

每一条知识是一个 `Knowledge` 对象，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` | 知识来源标识，如 `"physics"`、`"thermodynamics"`、`"my_task"` |
| `key` | `str` | **检索键**，描述构成新符号所需的**原材料符号**。LLM 会根据当前符号与 `key` 的语义相似度进行检索。建议包含符号名称和物理含义，用 `$...$` 包裹符号名。 |
| `target` | `str` | **目标符号**，描述要生成的新符号。同样建议用 `$...$` 包裹。 |
| `content` | `str` | **核心内容**，详细说明如何组合 `key` 中的符号得到 `target`。这是 LLM 学习的“组合方法”。 |
| `comment` | `str` | （可选）人类评价，如 `"Good symbol with excellent fitting performance"` |
| `reflection` | `str` | （可选）AI 反思总结，如 `"I should consider the relationship between speed and time headway"` |

### 示例代码

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge

# 1. 初始化 Agent（会自动创建/加载 Chroma 数据库）
agent = RAG_AGENT(
    port="7890",
    address="172.22.0.1",
    memory_path="codes/ragLibrary/memory_thermodynamics",
    fewshot_num=2,
    reflection_num=1,
    extend_num=1,
)

# 2. 添加一条知识：理想气体状态方程中 "pressure * volume" 的组合经验
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="thermodynamics",
        key=(
            r"$P$: The symbol in pressure unit which represents the pressure of the gas, "
            r"$V$: The symbol in volume unit which represents the volume of the container"
        ),
        target=(
            r"$PV$: The symbol in energy unit which represents the product of pressure and volume"
        ),
        content=(
            r"Human experts multiply $P$ with $V$ to obtain a new symbol $PV$, which is #mul P V#. "
            r"According to the ideal gas law, the product of pressure and volume is proportional to temperature, "
            r"so this combination has clear physical meaning and can be further used to construct $nRT$."
        ),
        comment="Good symbol that reflects the energy term in ideal gas law.",
        reflection="I need to consider multiplying pressure and volume when dealing with gas thermodynamics.",
    )
)

# 3. 保存目标名称索引（用于后续检索优化）
agent.knowledge_pool.save_target_names()

print("Current knowledge count:", len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings']))
```

### 添加多条知识的模板

如果你有**先验的公式结构知识**（例如已知某些物理量之间存在线性/乘积/比例关系），建议一次性写入：

```python
knowledge_list = [
    {
        "source": "my_domain",
        "key": "$x_1$: first input variable, $x_2$: second input variable",
        "target": "$x_1x_2$: product of two inputs",
        "content": "Multiply $x_1$ and $x_2$ to capture interaction effect: #mul x_1 x_2#.",
        "comment": "Interaction term is essential in this system.",
        "reflection": "Always try product of inputs when interaction is expected.",
    },
    # ... more knowledge
]

for k in knowledge_list:
    agent.knowledge_pool.add_knowledge(Knowledge(**k))

agent.knowledge_pool.save_target_names()
```

---

## 5. 删除与清空知识库

```python
# 删除指定索引的知识（注意：索引对应 Chroma 内部顺序）
agent.knowledge_pool.delete_knowledge([0, 1, 2])

# 清空整个知识库
knowledge_length = len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])
for _ in range(knowledge_length):
    agent.knowledge_pool.delete_knowledge([0])
```

---

## 6. 在通用符号回归中使用 RAG

在调用 `general_symbolic_regression` 时，设置 `use_rag=True` 并传入 `memory_path`：

```python
from codes.applications.general_symbolic_regression import general_symbolic_regression

expr, func = general_symbolic_regression(
    X, y,
    variable_names=["P", "V", "n", "T"],
    variable_units=[[1, -2, 1, 0, 0, 0, 0],   # pressure: [m, s, kg, ...] -> kg/(m·s²)
                   [3, 0, 0, 0, 0, 0, 0],    # volume: m³
                   [0, 0, 0, 0, 0, 0, 1],    # mole: mol
                   [0, 0, 0, 1, 0, 0, 0]],   # temperature: K
    variable_descriptions=["pressure", "volume", "amount of substance", "temperature"],
    target_name="E",
    target_unit=[2, -2, 1, 0, 0, 0, 0],  # energy: kg·m²/s²
    target_description="internal energy",
    use_rag=True,
    memory_path="codes/ragLibrary/memory_thermodynamics",
    n_epochs=50,
    n_evolutions=10,
)
```

> **提示**：如果当前任务完全没有先验知识，可以先用 `use_rag=False` 运行一次，让框架通过 Reflection 自动生成知识并写入 RAG 库；后续运行再开启 `use_rag=True` 复用这些知识。

---

## 7. 构建高质量 RAG 知识库的建议

1. **知识粒度要适中**：一条知识描述**一个**组合步骤（如 `a + b` 或 `sin(c)`），不要一次性描述整个公式。LLM 通过组合多条知识来构建复杂表达式。

2. **key 要包含物理含义**：不要只写 `$x_1$, $x_2$`，而要写 `$x_1$: speed of the object, $x_2$: time elapsed`。这样语义检索才能匹配到正确的上下文。

3. **content 中使用 `#...#` 标记前缀表达式**：虽然 LLM 主要学习自然语言描述，但在 `content` 中用 `#mul P V#` 等形式显式标注组合方式，有助于人类检查和调试。

4. **comment + reflection 闭环**：`comment` 是人类评价，`reflection` 是 AI 应该从这条知识中学到的**通用规律**。高质量的 reflection 能让 LLM 在相似场景中举一反三。

5. **分领域维护不同的 memory_path**：不要在同一个库中混存汽车跟驰、热力学、流体力学的知识，否则检索时会引入噪声。建议按领域拆分：
   - `memory_car_following`
   - `memory_thermodynamics`
   - `memory_fluid_dynamics`
   - `memory_my_paper_experiment`

---

## 8. 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `Embedding error, retrying...` | OpenAI API 调用失败 | 检查 `OPENAI_API_KEY` 和代理地址/端口 |
| 检索到的知识完全不相关 | `key` 写得太简略 | 在 `key` 中加入更多物理含义描述 |
| LLM 生成的符号重复或无效 | RAG 库中知识太少或质量低 | 先运行 `use_rag=False` 收集 reflection，或手动添加高质量知识 |
| `chroma.sqlite3` 锁定错误 | 多进程同时访问 | 确保同一时刻只有一个进程读写同一个 `memory_path` |

---

## 9. 参考文件

- 核心 RAG 类：`codes/trafficSR/D_updation_by_LLM/RAG_Agent.py`
- 知识池实现：`codes/trafficSR/D_updation_by_LLM/Modules/Knowledge_Pool.py`
- Prompt 生成器：`codes/trafficSR/D_updation_by_LLM/Modules/Prompt_Generator.py`
- 通用符号回归入口：`codes/applications/general_symbolic_regression.py`
- 已有知识库示例：`codes/dependencies/generate_rag_library/generate_new_formula/operate_knowledge_find_new.py`
