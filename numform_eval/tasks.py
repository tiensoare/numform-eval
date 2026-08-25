"""
Evaluation tasks: math problems whose inputs are Numbers and whose answers
we can compute exactly, so a model response can be scored.

Each Task knows:
  - the Numbers it involves
  - how to build the prompt given a chosen representation method
  - the exact ground-truth answer (as a sympy value)
  - how to check a model's numeric answer against ground truth

Keeping ground truth symbolic (via exact_expr) means the "correct" answer
does not itself suffer the truncation we're studying.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import sympy as sp

from .transforms import Number, render


@dataclass
class Task:
    task_id: str
    kind: str                       # "arithmetic", "compare", "reasoning", ...
    numbers: List[Number]
    prompt_template: str            # uses {0}, {1}, ... for rendered numbers
    ground_truth: sp.Expr           # exact answer
    answer_tol: float = 1e-4        # tolerance when scoring the model's float answer

    def build_prompt(self, method: str, **kwargs) -> str:
        rendered = [render(n, method, **kwargs) for n in self.numbers]
        body = self.prompt_template.format(*rendered)
        # A strict output contract makes parsing reliable across models.
        return (
            body
            + "\n\nThink step by step, then give the final numeric answer on the "
            + "last line in exactly this format:\nANSWER: <number>"
        )

    def truth_float(self) -> float:
        return float(self.ground_truth)


def default_tasks() -> List[Task]:
    """A small starter set spanning a few task kinds."""
    pi = sp.pi
    sqrt2 = sp.sqrt(2)

    n_pi = Number("pi", float(pi), pi)
    n_third = Number("1/3", 1/3, sp.Rational(1, 3))
    n_sqrt2 = Number("sqrt(2)", float(sqrt2), sqrt2)
    n_two = Number("2", 2.0, sp.Integer(2))
    n_seventh = Number("1/7", 1/7, sp.Rational(1, 7))
    n_piq = Number("pi/4", float(pi/4), pi/4)

    return [
        Task(
            task_id="mul_pi_third",
            kind="arithmetic",
            numbers=[n_pi, n_third],
            prompt_template="Compute {0} * {1}.",
            ground_truth=pi * sp.Rational(1, 3),
        ),
        Task(
            task_id="add_sqrt2_third",
            kind="arithmetic",
            numbers=[n_sqrt2, n_third],
            prompt_template="Compute {0} + {1}.",
            ground_truth=sqrt2 + sp.Rational(1, 3),
        ),
        Task(
            task_id="square_sqrt2",
            kind="reasoning",
            numbers=[n_sqrt2],
            prompt_template="Compute {0} squared.",
            ground_truth=sp.Integer(2),   # sqrt(2)^2 = 2 exactly; decimals drift
        ),
        Task(
            task_id="compare_pi_seventh",
            kind="compare",
            numbers=[n_pi, n_seventh],
            prompt_template=(
                "Which is larger, {0} or {1}? "
                "Answer 1 if the first is larger, 0 otherwise."
            ),
            ground_truth=sp.Integer(1),
        ),
        Task(
            task_id="mul_seventh_seven",
            kind="arithmetic",
            numbers=[n_seventh, n_two],
            prompt_template="Compute {0} * {1}.",
            ground_truth=sp.Rational(2, 7),
        ),
        Task(
            task_id="four_times_piq",
            kind="reasoning",
            numbers=[n_piq],
            prompt_template="Compute 4 * {0}.",
            ground_truth=pi,   # 4 * pi/4 = pi; symbolic form should nail it
        ),
    ]


if __name__ == "__main__":
    for t in default_tasks():
        print(f"[{t.task_id}] ({t.kind}) truth = {t.ground_truth} = {t.truth_float():.6f}")
        for m in ("original", "limit", "nsimplify"):
            print(f"   {m:<10}: {t.build_prompt(m).splitlines()[0]}")
        print()
