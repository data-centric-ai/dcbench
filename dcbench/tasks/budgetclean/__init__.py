# flake8: noqa

from ...common import Task
from ...common.bundle import RelationalBundle
from .common import Preprocessor
from .problem import BudgetcleanProblem, BudgetcleanSolution
from .baselines import random_clean, cp_clean

# TODO: consider changing this name, word minimal doesn't feel appropriate since
# the budget is fixed
task = Task(
    task_id="budgetclean",
    name="Minimal Feature Cleaning",
    summary=(
        "When it comes to data preparation, data cleaning is often an essential yet "
        "quite costly task. If we are given a fixed cleaning budget, the challenge is "
        "to find the training data examples that would would bring the biggest "
        "positive impact on model performance if we were to clean them."
    ),
    problem_class=BudgetcleanProblem,
    solution_class=BudgetcleanSolution,
    baselines=RelationalBundle(
        items={"random_clean": random_clean, "cp_clean": cp_clean},
        attributes=["summary"]
    )
)
