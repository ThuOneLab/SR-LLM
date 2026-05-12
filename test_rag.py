"""
Test RAG (Retrieval-Augmented Generation) for symbolic regression.

We build a small RAG knowledge pool with common polynomial construction
patterns, then test if it helps discover y = x1^2 + 2*x2 + 1.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge
from codes.applications.general_symbolic_regression import general_symbolic_regression


def build_rag_library(memory_path="codes/ragLibrary/memory_polynomial"):
    """Build a small RAG library with polynomial construction knowledge."""
    agent = RAG_AGENT(
        port="7890",
        address="172.22.0.1",
        memory_path=memory_path,
        fewshot_num=2,
        reflection_num=1,
        extend_num=1,
    )

    # Clear existing knowledge (if any)
    knowledge_length = len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])
    for _ in range(knowledge_length):
        agent.knowledge_pool.delete_knowledge([0])

    # Knowledge 1: How to construct a square term
    agent.knowledge_pool.add_knowledge(
        Knowledge(
            source="polynomial",
            key=(
                r"$x_1$: The symbol in dimensionless unit which represents the first input variable"
            ),
            target=(
                r"$x_1^2$: The symbol in dimensionless unit which represents the square of the first input variable"
            ),
            content=(
                r"To obtain the square of a variable, human experts apply the square operator to the variable. "
                r"For example, squaring $x_1$ gives a new symbol $x_1^2$, which is #n2(x_1)#. "
                r"This operation preserves the dimensionless nature of the original variable."
            ),
            comment="Good symbol for constructing polynomial terms involving squares.",
            reflection="When encountering a variable, consider squaring it to capture quadratic relationships.",
        )
    )

    # Knowledge 2: How to construct a linear interaction term
    agent.knowledge_pool.add_knowledge(
        Knowledge(
            source="polynomial",
            key=(
                r"$x_2$: The symbol in dimensionless unit which represents the second input variable, "
                r"$c$: The symbol in dimensionless unit which represents a free constant"
            ),
            target=(
                r"$c \cdot x_2$: The symbol in dimensionless unit which represents the second input variable scaled by a constant"
            ),
            content=(
                r"To scale a variable by a constant coefficient, human experts multiply the variable with a free constant. "
                r"For example, multiplying $x_2$ with constant $c$ gives #mul(c, x_2)#. "
                r"This allows fitting linear terms with adjustable coefficients."
            ),
            comment="Good symbol for constructing linear terms with learnable coefficients.",
            reflection="When a linear term with unknown coefficient is needed, multiply the variable with a free constant.",
        )
    )

    # Knowledge 3: How to combine terms into a polynomial
    agent.knowledge_pool.add_knowledge(
        Knowledge(
            source="polynomial",
            key=(
                r"$x_1^2$: The symbol in dimensionless unit which represents the square of the first input variable, "
                r"$c \cdot x_2$: The symbol in dimensionless unit which represents the second input variable scaled by a constant"
            ),
            target=(
                r"$x_1^2 + c \cdot x_2 + 1$: The symbol in dimensionless unit which represents a quadratic polynomial"
            ),
            content=(
                r"To construct a polynomial, human experts add multiple terms together. "
                r"For example, adding the square term $x_1^2$, the linear term $c \cdot x_2$, and a constant 1 gives "
                r"#add(add(n2(x_1), mul(c, x_2)), 1)#. "
                r"This combination captures both quadratic and linear relationships."
            ),
            comment="Good symbol for constructing complete polynomial expressions.",
            reflection="When building a polynomial, combine power terms, linear terms, and constants through addition.",
        )
    )

    agent.knowledge_pool.save_target_names()
    print(f"RAG library built at {memory_path} with {len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])} pieces of knowledge.")
    return memory_path


if __name__ == "__main__":
    # Step 1: Build RAG library
    print("=" * 60)
    print("Step 1: Building RAG knowledge library...")
    print("=" * 60)
    memory_path = build_rag_library("codes/ragLibrary/memory_polynomial")

    # Step 2: Generate data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.rand(n_samples, 2) * 10
    y = X[:, 0] ** 2 + 2 * X[:, 1] + 1

    # Step 3: Run symbolic regression WITH RAG
    print("\n" + "=" * 60)
    print("Step 2: Running symbolic regression WITH RAG...")
    print("=" * 60)

    best_expr, best_func = general_symbolic_regression(
        X,
        y,
        variable_names=["x1", "x2"],
        variable_units=[[0, 0], [0, 0]],
        variable_descriptions=["first input variable", "second input variable"],
        target_name="y",
        target_unit=[0, 0],
        target_description="output variable",
        seed=100,
        n_epochs=10,
        n_evolutions=2,
        use_rag=True,
        memory_path=memory_path,
        device="cpu",
    )

    print("\n" + "=" * 60)
    print("TARGET:  y = x1**2 + 2*x2 + 1")
    print("FOUND:  ", best_expr)
    print("=" * 60)
