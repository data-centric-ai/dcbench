from typing import Any, Mapping

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from dcbench.common.artefact import ArtefactSpec, CSVArtefact
from dcbench.common.problem import Problem
from dcbench.common.solution import Result, Solution

from .common import Preprocessor


class MiniCleanSolution(Solution):
    artefact_specs: Mapping[str, ArtefactSpec] = {
        "train_ids": ArtefactSpec(artefact_type=CSVArtefact, description="")
    }


class MinicleanProblem(Problem):

    # TODO: consider changing this name, word minimal doesn't feel appropriate since
    # the budget is fixed
    full_name = "Minimal Feature Cleaning"

    summary = (
        "When it comes to data preparation, data cleaning is often an essential yet "
        "quite costly task. If we are given a fixed cleaning budget, the challenge is "
        "to find the training data examples that would would bring the biggest "
        "positive impact on model performance if we were to clean them."
    )

    artefact_specs: Mapping[str, ArtefactSpec] = {
        "X_train_dirty_a": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "X_train_dirty_b": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "X_train_clean_a": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "X_train_clean_b": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "y_train_a": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "y_train_b": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "X_val": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "y_val": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "X_test": ArtefactSpec(artefact_type=CSVArtefact, description=""),
        "y_test": ArtefactSpec(artefact_type=CSVArtefact, description=""),
    }

    task_id: str = "miniclean"

    solution_class: type = MiniCleanSolution

    @classmethod
    def list(cls):
        for scenario_id in cls.scenario_df["id"]:
            yield cls.from_id(scenario_id)

    @classmethod
    def from_id(cls, scenario_id: str):
        pass

    def solve(self, **kwargs: Any) -> Solution:

        if len(kwargs) not in [1, 2]:
            raise ValueError("The solution can be built from either 1 or 2 objects.")

        if not all(isinstance(x, pd.DataFrame) for x in kwargs.values()):
            raise ValueError("The solution objects must be Pandas DataFrame instances.")

        if len(self.artefacts["X_train_dirty_a"].load()) != len(
            kwargs["X_train_selection_a"]
        ):
            raise ValueError(
                "The first data frame must have the same size as X_train_dirty_a."
            )
        if len(kwargs) == 2 and len(self.artefacts["X_train_dirty_b"].load()) != len(
            kwargs["X_train_selection_b"]
        ):
            raise ValueError(
                "The second data frame must have the same size as X_train_dirty_a."
            )

        if not all(len(item.columns) == 1 for item in kwargs.values()):
            raise ValueError(
                "All provided solution objects must be made up of a single column."
            )
        if not all(item.dtypes[0] == bool for item in kwargs.values()):
            raise ValueError("All provided solution objects must have boolean columns.")

        return Solution(self, objects=kwargs)

    def evaluate(self, solution: Solution) -> "Result":

        # Load scenario artefacts.
        a = self.artefacts.load()

        # Load solution artefacts.
        I_solution_a = solution.artefacts.X_train_selection_a.load()
        I_solution_b = (
            solution.artefacts.X_train_selection_b.load()
            if len(solution.artefacts) > 1
            else None
        )

        # Determine the solution training datasets.
        X_train_solution_a = a.X_train_dirty_a.mask(I_solution_a, a.X_train_clean_a)
        X_train_solution_b = (
            a.X_train_dirty_b.mask(I_solution_b, a.X_train_clean_b)
            if I_solution_b is not None
            else None
        )

        # Fit data preprocessor.
        preprocessor = Preprocessor()
        X_train_dirty = pd.concat(
            [a.X_train_dirty_a, a.X_train_dirty_b], ignore_index=True, sort=False
        )
        y_train = pd.concat([a.y_train_a, a.y_train_b], ignore_index=True, sort=False)
        preprocessor.fit(X_train_dirty, y_train)

        # Preprocess the data.
        X_train_solution_a, y_train_a = preprocessor.transform(
            X_train_solution_a, a.y_train_a
        )
        if X_train_solution_b is not None:
            X_train_solution_b, y_train_b = preprocessor.transform(
                X_train_solution_b, a.y_train_b
            )
        X_val, y_val = preprocessor.transform(a.X_val, a.y_val)
        if a.X_test is not None and a.y_test is not None:
            X_test, y_test = preprocessor.transform(a.X_test, a.y_test)

        # Train the model.
        model_a = KNeighborsClassifier(n_neighbors=3).fit(X_train_solution_a, y_train_a)
        if X_train_solution_b is not None:
            model_b = KNeighborsClassifier(n_neighbors=3).fit(
                X_train_solution_b, y_train_b
            )

        # Evaluate the model.
        result = {}
        result["acc_a_val"] = model_a.score(X_val, y_val)
        if X_train_solution_b is not None:
            result["acc_b_val"] = model_b.score(X_val, y_val)
        if a.X_test is not None and a.y_test is not None:
            result["acc_a_test"] = model_a.score(X_test, y_test)
        if (
            X_train_solution_b is not None
            and a.X_test is not None
            and a.y_test is not None
        ):
            result["acc_b_test"] = model_b.score(X_test, y_test)
        return result
