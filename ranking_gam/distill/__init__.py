"""PWL distillation (Paper Algorithm 1)."""

from .pwl import (
    greedy_knot_selection,
    distill_to_pwl,
    distill_context_model,
    pwl_predict,
    evaluate_pwl,
)

__all__ = [
    "greedy_knot_selection",
    "distill_to_pwl",
    "distill_context_model",
    "pwl_predict",
    "evaluate_pwl",
]
