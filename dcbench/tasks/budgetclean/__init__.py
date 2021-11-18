# flake8: noqa

from ...common import Task
from .common import Preprocessor
from .problem import BudgetCleanProblem, BudgetCleanSolution

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
    problem_class=BudgetCleanProblem,
    solution_class=BudgetCleanSolution,
)
