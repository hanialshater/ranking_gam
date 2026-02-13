"""PWL distillation (Paper Algorithm 1)."""

from .pwl import (
    distill_context_model,
    distill_to_pwl,
    evaluate_pwl,
    greedy_knot_selection,
    pwl_predict,
)

__all__ = [
    "greedy_knot_selection",
    "distill_to_pwl",
    "distill_context_model",
    "pwl_predict",
    "evaluate_pwl",
]
