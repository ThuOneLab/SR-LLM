# RAG Library Construction and Usage Guide

This document explains how to build and maintain a RAG (Retrieval-Augmented Generation) knowledge pool for **SR-LLM general symbolic regression** (`general_symbolic_regression.py`).

---

## 1. What is a RAG Knowledge Pool?

In the SR-LLM framework, a RAG knowledge pool is a **Chroma**-based vector database that stores **expert knowledge** about "how to combine symbols to form new symbols". In each Evolution round, the LLM will:

1. Retrieve similar historical knowledge from the RAG pool based on the current best-performing symbols;
2. Refer to the combination methods in that knowledge to generate new symbol combinations;
3. Perform **Reflection** on excellent expressions and write the newly learned experience back into the RAG pool.

Therefore, a high-quality RAG knowledge pool can significantly improve the LLM's ability to generate effective symbols and accelerate the convergence of symbolic regression.

---

## 2. Prerequisites

- SR-LLM environment installed (`conda activate sr-llm`)
- OpenAI API Key configured:
  ```bash
  export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
  ```
- If used in mainland China, a proxy address is recommended (default `172.22.0.1:7890`, modifiable in code)

---

## 3. Initializing a RAG Knowledge Pool

A RAG knowledge pool is essentially a local folder plus a SQLite file. It is **created automatically** on first run; no manual table creation is needed.

```python
from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT

# Specify the storage path (relative to project root)
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

After running, the following files will be generated under `codes/ragLibrary/memory_my_task/`:
- `chroma.sqlite3` — Chroma vector database file
- `knowledge_targets_names.json` — Knowledge target name index (auto-maintained)

---

## 4. Adding Knowledge

Each piece of knowledge is a `Knowledge` object containing the following fields:

| Field | Type | Description |
|------|------|-------------|
| `source` | `str` | Knowledge source identifier, e.g., `"physics"`, `"thermodynamics"`, `"my_task"` |
| `key` | `str` | **Retrieval key**, describing the **raw material symbols** needed to construct the new symbol. The LLM retrieves based on semantic similarity between the current symbol and the `key`. It is recommended to include symbol names and physical meanings, wrapping symbol names in `$...$`. |
| `target` | `str` | **Target symbol**, describing the new symbol to be generated. Also recommended to wrap in `$...$`. |
| `content` | `str` | **Core content**, detailing how to combine the symbols in `key` to obtain `target`. This is the "combination method" that the LLM learns. |
| `comment` | `str` | (Optional) Human evaluation, e.g., `"Good symbol with excellent fitting performance"` |
| `reflection` | `str` | (Optional) AI reflection summary, e.g., `"I should consider the relationship between speed and time headway"` |

### Example Code

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge

# 1. Initialize the Agent (will auto-create/load the Chroma database)
agent = RAG_AGENT(
    port="7890",
    address="172.22.0.1",
    memory_path="codes/ragLibrary/memory_thermodynamics",
    fewshot_num=2,
    reflection_num=1,
    extend_num=1,
)

# 2. Add a piece of knowledge: combination experience of "pressure * volume" from the ideal gas law
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

# 3. Save the target name index (used for retrieval optimization)
agent.knowledge_pool.save_target_names()

print("Current knowledge count:", len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings']))
```

### Template for Adding Multiple Pieces of Knowledge

If you have **prior formula structure knowledge** (e.g., knowing that certain physical quantities have linear/product/proportional relationships), it is recommended to write them in batch:

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

## 5. Deleting and Clearing the Knowledge Pool

```python
# Delete knowledge at specified indices (note: indices correspond to Chroma internal order)
agent.knowledge_pool.delete_knowledge([0, 1, 2])

# Clear the entire knowledge pool
knowledge_length = len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])
for _ in range(knowledge_length):
    agent.knowledge_pool.delete_knowledge([0])
```

---

## 6. Using RAG in General Symbolic Regression

When calling `general_symbolic_regression`, set `use_rag=True` and pass the `memory_path`:

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

> **Tip**: If the current task has no prior knowledge at all, you can first run with `use_rag=False` once, letting the framework automatically generate knowledge through Reflection and write it into the RAG pool; in subsequent runs, enable `use_rag=True` to reuse this knowledge.

---

## 7. Recommendations for Building a High-Quality RAG Knowledge Pool

1. **Moderate knowledge granularity**: Each piece of knowledge should describe **one** combination step (e.g., `a + b` or `sin(c)`). Do not describe an entire formula at once. The LLM builds complex expressions by combining multiple pieces of knowledge.

2. **The `key` should contain physical meaning**: Do not just write `$x_1$, $x_2$`; instead write `$x_1$: speed of the object, $x_2$: time elapsed`. This way semantic retrieval can match the correct context.

3. **Use `#...#` to mark prefix expressions in `content`**: Although the LLM mainly learns from natural language descriptions, explicitly annotating combination methods like `#mul P V#` in `content` helps human inspection and debugging.

4. **Close the loop with `comment` + `reflection`**: `comment` is human evaluation, and `reflection` is the **general rule** that the AI should learn from this piece of knowledge. High-quality reflections enable the LLM to generalize to similar scenarios.

5. **Maintain separate `memory_path` per domain**: Do not mix car-following, thermodynamics, and fluid dynamics knowledge in the same pool, as this introduces noise during retrieval. It is recommended to split by domain:
   - `memory_car_following`
   - `memory_thermodynamics`
   - `memory_fluid_dynamics`
   - `memory_my_paper_experiment`

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `Embedding error, retrying...` | OpenAI API call failed | Check `OPENAI_API_KEY` and proxy address/port |
| Retrieved knowledge is completely irrelevant | `key` is too brief | Add more physical meaning descriptions in the `key` |
| LLM generates duplicate or invalid symbols | Too little or low-quality knowledge in the RAG pool | First run with `use_rag=False` to collect reflections, or manually add high-quality knowledge |
| `chroma.sqlite3` lock error | Multiple processes accessing simultaneously | Ensure only one process reads/writes the same `memory_path` at a time |

---

## 9. Reference Files

- Core RAG class: `codes/trafficSR/D_updation_by_LLM/RAG_Agent.py`
- Knowledge pool implementation: `codes/trafficSR/D_updation_by_LLM/Modules/Knowledge_Pool.py`
- Prompt generator: `codes/trafficSR/D_updation_by_LLM/Modules/Prompt_Generator.py`
- General symbolic regression entry: `codes/applications/general_symbolic_regression.py`
- Existing knowledge pool example: `codes/dependencies/generate_rag_library/generate_new_formula/operate_knowledge_find_new.py`
