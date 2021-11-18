import warnings
from typing import Any, Mapping

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dcbench.common import Problem, Result, Solution
from dcbench.common.artifact import ArtifactSpec, CSVArtifact

from .common import Preprocessor


class BudgetcleanSolution(Solution):
    artifact_specs: Mapping[str, ArtifactSpec] = {
        "idx_selected": ArtifactSpec(artifact_type=CSVArtifact, description="")
    }


class BudgetcleanProblem(Problem):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "X_train_dirty": ArtifactSpec(
            artifact_type=CSVArtifact,
            description=(
                "Features of the dirty training dataset which we need to clean. "
                "Each dirty cell contains an embedded list of clean "
                "candidate values.",
            ),
        ),
        "X_train_clean": ArtifactSpec(
            artifact_type=CSVArtifact,
            description="Features of the clean training dataset where each dirty value "
            "from the dirty dataset is replaced with the correct "
            "clean candidate.",
        ),
        "y_train": ArtifactSpec(
            artifact_type=CSVArtifact, description="Labels of the training dataset."
        ),
        "X_val": ArtifactSpec(
            artifact_type=CSVArtifact,
            description="Feature of the validtion dataset which can be used to guide "
            "the cleaning optimization process.",
        ),
        "y_val": ArtifactSpec(
            artifact_type=CSVArtifact, description="Labels of the validation dataset."
        ),
        "X_test": ArtifactSpec(
            artifact_type=CSVArtifact,
            description=(
                "Features of the test dataset used to produce the final evaluation "
                "score of the model.",
            ),
        ),
        "y_test": ArtifactSpec(
            artifact_type=CSVArtifact, description="Labels of the test dataset."
        ),
    }

    task_id: str = "budgetclean"

    @classmethod
    def list(cls):
        for scenario_id in cls.scenario_df["id"]:
            yield cls.from_id(scenario_id)

    @classmethod
    def from_id(cls, scenario_id: str):
        pass

    def solve(self, idx_selected: Any, **kwargs: Any) -> Solution:

        # Construct the solution object as a Pandas DataFrame.
        idx_selected_df = None
        if isinstance(idx_selected, pd.DataFrame):
            idx_selected_df = pd.DataFrame(
                {"idx_selected": idx_selected.iloc[:, 0].values}
            ).astype("bool")
        elif isinstance(idx_selected, list):
            idx_selected_df = pd.DataFrame({"idx_selected": idx_selected}).astype(
                "bool"
            )
        else:
            raise ValueError(
                "The provided idx_selected object must be either a list or a DataFrame."
            )

        # Check if the content of the solution object is valid.
        X_train_dirty = self["X_train_dirty"]
        if len(X_train_dirty) != len(idx_selected_df):
            raise ValueError(
                "The number of elements of the provided solution object must be the "
                "same as for the training dataset. (expected: %d, found: %d)"
                % (len(X_train_dirty), len(idx_selected_df))
            )

        num_selected = idx_selected_df["idx_selected"].sum()
        budget = int(self.attributes["budget"] * len(X_train_dirty))
        if num_selected > budget:
            raise ValueError(
                "The number of selected data examples is  "
                "higher than the allowed budget. "
                "(expected: %d, found: %d)" % (budget, num_selected)
            )
        if num_selected < budget:
            warnings.warn(
                "The number of selected data examples is below the allowed budget. "
                "(expected: %d, found: %d)" % (budget, num_selected)
            )

        # Construct and return a solution object.
        solution = BudgetcleanSolution.from_artifacts({"idx_selected": idx_selected_df})
        solution.attributes["problem_id"] = self.container_id
        for k, v in self.attributes.items():
            solution.attributes[k] = v
        return solution

    def evaluate(self, solution: BudgetcleanSolution) -> "Result":

        # Load scenario artifacts.
        X_train_dirty = self["X_train_dirty"]
        X_train_clean = self["X_train_clean"]
        y_train = self["y_train"]
        X_val = self["X_val"]
        y_val = self["y_val"]
        X_test = self["X_test"]
        y_test = self["y_test"]

        # Replace lists with None values.
        def clearlists(x):
            if isinstance(x, list):
                return None
            return x

        X_train_dirty = X_train_dirty.applymap(clearlists)

        # Load solution artifacts.
        idx_selected = solution["idx_selected"]["idx_selected"]

        # Determine the solution training datasets.
        X_train_solution = X_train_dirty.mask(idx_selected, X_train_clean)

        # Fit data preprocessor.
        preprocessor = Preprocessor()
        preprocessor.fit(X_train_dirty, y_train)

        # Preprocess the data.
        X_train_solution, y_train = preprocessor.transform(X_train_solution, y_train)
        X_train_dirty = preprocessor.transform(X_train_dirty)
        X_train_clean = preprocessor.transform(X_train_clean)
        X_val, y_val = preprocessor.transform(X_val, y_val)
        X_test, y_test = preprocessor.transform(X_test, y_test)

        # Train the solution, clean and dirty models.
        if self.attributes["model"] == "logreg":
            model_solution = LogisticRegression().fit(X_train_solution, y_train)
            model_dirty = LogisticRegression().fit(X_train_dirty, y_train)
            model_clean = LogisticRegression().fit(X_train_clean, y_train)
        elif self.attributes["model"] == "randomf":
            model_solution = RandomForestClassifier().fit(X_train_solution, y_train)
            model_dirty = RandomForestClassifier().fit(X_train_dirty, y_train)
            model_clean = RandomForestClassifier().fit(X_train_clean, y_train)
        else:
            raise ValueError("Unknown model attribute '%s'." % self.attributes["model"])

        # Evaluate the model.
        result_dict = {}
        acc_val_solution = model_solution.score(X_val, y_val)
        acc_val_dirty = model_dirty.score(X_val, y_val)
        acc_val_clean = model_clean.score(X_val, y_val)
        result_dict["acc_val_gapclosed"] = (acc_val_solution - acc_val_dirty) / (
            acc_val_clean - acc_val_dirty
        )
        acc_test_solution = model_solution.score(X_test, y_test)
        acc_test_dirty = model_dirty.score(X_test, y_test)
        acc_test_clean = model_clean.score(X_test, y_test)
        result_dict["acc_test_gapclosed"] = (acc_test_solution - acc_test_dirty) / (
            acc_test_clean - acc_test_dirty
        )

        result_dict = {**result_dict, **solution.attributes}

        return pd.Series(result_dict, name=solution.container_id)
