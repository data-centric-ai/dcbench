# flake8: noqa

from ...common import Task
from ...common.table import Table
from .baselines import cp_clean, random_clean
from .common import Preprocessor
from .problem import BudgetcleanProblem, BudgetcleanSolution

__all__ = [""]


task = Task(
    task_id="budgetclean",
    name="Data Cleaning on a Budget ",
    summary=(
        "When it comes to data preparation, data cleaning is an essential yet "
        "quite costly task. If we are given a fixed cleaning budget, the challenge is "
        "to find the training data examples that would would bring the biggest "
        "positive impact on model performance if we were to clean them."
    ),
    problem_class=BudgetcleanProblem,
    solution_class=BudgetcleanSolution,
    baselines=Table(
        data=[random_clean, cp_clean],
        # attributes=["summary"],
    ),
)
