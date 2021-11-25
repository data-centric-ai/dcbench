import meerkat as mk
import os
import pandas as pd
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from dcbench.common import Problem, Solution, Task
from dcbench.common.artifact import ArtifactSpec, DataPanelArtifact, YAMLArtifact


class MiniDataSolution(Solution):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "train_ids": ArtifactSpec(
            artifact_type=YAMLArtifact,
            description=(
                "A list of train example ids from the "
                " ``id`` column of ``train_data``."
            ),
        ),
    }
    task_id: str = "minidata"

    @classmethod
    def from_ids(cls, train_ids: Sequence[str], problem_id: str):
        cls.from_artifacts(
            {"train_ids": train_ids}, attributes={"problem_id": problem_id}
        )


class MiniDataProblem(Problem):

    artifact_specs: Mapping[str, ArtifactSpec] = {
        "train_data": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of train examples with columns ``id``, "
            "``input``, and ``target``.",
        ),
        "val_data": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of validation examples with columns ``id``, "
            "``input``, and ``target``.",
        ),
        "test_data": ArtifactSpec(
            artifact_type=DataPanelArtifact,
            description="A DataPanel of test examples with columns ``id``, "
            "``input``, and ``target``.",
        ),
    }

    task_id: str = "minidata"


    def solve(self, idx_selected: Any, **kwargs: Any) -> Solution:

        # Construct the solution object as a Pandas DataFrame.
        idx_selected_dp = None
        if isinstance(idx_selected, mk.DataPanel):
            idx_selected_dp = mk.DataPanel(
                {"idx_selected": idx_selected[idx_selected.columns[0]].data.astype(bool)}
            )
        elif isinstance(idx_selected, pd.DataFrame):
            idx_selected_dp = mk.DataPanel(
                {"idx_selected": idx_selected.iloc[:, 0].values.astype(bool)}
            )
        elif isinstance(idx_selected, list):
            idx_selected_dp = mk.DataPanel({"idx_selected": idx_selected}).astype(
                "bool"
            )
        else:
            raise ValueError(
                "The provided idx_selected object must be either a list or a DataFrame."
            )

        # Check if the content of the solution object is valid.
        X_train_dirty = self["X_train_dirty"]
        if len(X_train_dirty) != len(idx_selected_dp):
            raise ValueError(
                "The number of elements of the provided solution object must be the "
                "same as for the training dataset. (expected: %d, found: %d)"
                % (len(X_train_dirty), len(idx_selected_dp))
            )

        # Construct and return a solution object.
        solution = MiniDataSolution.from_artifacts({"idx_selected": idx_selected_dp})
        solution.attributes["problem_id"] = self.container_id
        for k, v in self.attributes.items():
            solution.attributes[k] = v
        return solution


    def evaluate(self, solution: Solution):
        train_dp = self["train_data"]
        train_ids = solution["train_ids"]

        train_dp = train_dp.lz[train_dp["id"].isin(train_ids)]
        dirpath = tempfile.mkdtemp()
        dp_path = os.path.join(dirpath, "dataset.mk")

        train_dp.write(dp_path)

        from unagi.unagi import main
        from unagi.utils.config_utils import build_config

        from .unagi_configs import RESNET_CONFIG

        config = RESNET_CONFIG.copy()
        config["dataset"]["path_to_dp"] = dp_path
        config["dataset"]["index_name"] = "id"
        config = build_config(config)

        main(config)
        shutil.rmtree(dirpath)

        # TODO: Plug unagi in here
        # model = fit(train_dp)
        # score = score(self["test_dp"], model)
        # returnscore


task = Task(
    task_id="minidata",
    name="Minimal Data Selection",
    # flake8: noqa
    summary="Given a large training dataset, what is the smallest subset you can sample that still achieves some threshold of performance.",
    problem_class=MiniDataProblem,
    solution_class=MiniDataSolution,
    baselines=None,
)
