
from __future__ import annotations
from contextlib import redirect_stdout
import dataclasses
from gettext import dpgettext
import io
import itertools

from random import choice, sample
from typing import Collection, Dict, Iterable, List, Mapping, Tuple, Union
from dataclasses import dataclass
from domino import embed
from sklearn.linear_model import LinearRegression

import pandas as pd
from scipy.stats import rankdata
import numpy as np
import meerkat as mk
from tqdm.auto import tqdm
import os

from domino.utils import unpack_args
from dcbench import Artifact
from dcbench import SliceDiscoveryProblem, SliceDiscoverySolution
import dcbench
from .metrics import compute_solution_metrics

task = dcbench.tasks["slice_discovery"]

def _run_sdms(problems: List[SliceDiscoveryProblem], **kwargs):
    result = []
    for problem in problems:
        #f = io.StringIO()
        #with redirect_stdout(f):
        result.append(run_sdm(problem, **kwargs))
    return result

def run_sdms(
    problems: List[SliceDiscoveryProblem],
    slicer_class: type,
    slicer_config: dict,
    encoder: str = "clip", 
    variant: str = "ViT-B/32",
    batch_size: int = 1,
    num_workers: int = 0,
):

    # prepare embeddings
    base_datasets = set([p.artifacts["base_dataset"].id for p in problems])
    embs = {}
    for base_dataset in base_datasets:
        dataset_artifact = dcbench.VisionDatasetArtifact(base_dataset)
        emb_artifact_id = f"common/embeddings/{base_dataset}/{encoder}-{variant.replace('/', '-')}"

        emb_artifact = dcbench.DataPanelArtifact(emb_artifact_id)
        if os.path.exists(emb_artifact.local_path):
            emb_dp = emb_artifact.load()
        else:
            dataset_artifact.download()
            emb_dp = embed(
                dataset_artifact.load(),
                input_col="image",
                encoder=encoder,
                variant=variant,
                device=0,
                num_workers=12
            )
            emb_dp["emb"] = emb_dp[f"{encoder}(image)"]
            emb_dp.remove_column(f"{encoder}(image)")
            emb_artifact = Artifact.from_data(emb_dp, artifact_id=emb_artifact_id)
        embs[base_dataset] = emb_dp

    if num_workers > 0:
        import ray

        ray.init()
        run_fn = ray.remote(_run_sdms).remote
        embs = ray.put(embs) 
    else:
        run_fn = _run_sdms

    total_batches = len(problems)
    results = []
    t = tqdm(total=total_batches)

    for start_idx in range(0, len(problems), batch_size):
        batch = problems[start_idx : start_idx + batch_size]

        result = run_fn(
            problems=batch,
            embs=embs,
            slicer_class=slicer_class,
            slicer_config=slicer_config,
        )

        if num_workers == 0:
            t.update(n=len(result))
            results.extend(result)
        else:
            # in the parallel case, this is a single object reference
            # moreover, the remote returns immediately so we don't update tqdm
            results.append(result)

    if num_workers > 0:
        # if we're working in parallel, we need to wait for the results to come back
        # and update the tqdm accordingly
        result_refs = results
        results = []
        while result_refs:
            done, result_refs = ray.wait(result_refs)
            for result in done:
                result = ray.get(result)
                results.extend(result)
                t.update(n=len(result))
        ray.shutdown()
    solutions, metrics = zip(*results)
    # flatten the list of lists 
    metrics = [row for slices in metrics for row in slices]

    path = task.write_solutions(solutions)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(os.path.dirname(path), "metrics.csv"), index=False)

    return solutions, metrics_df


def run_sdm(
    problem: SliceDiscoveryProblem,
    slicer_class: type,
    slicer_config: dict,
    embs: Mapping[str, mk.DataPanel],
) -> SliceDiscoverySolution:
    emb_dp = embs[problem.artifacts["base_dataset"].id]
    val_dp = problem.merge(split="val")
    val_dp = val_dp.merge(emb_dp["id", "emb"], on="id", how="left")
    
    slicer = slicer_class(pbar=False, n_slices=problem.n_pred_slices, **slicer_config)
    slicer.fit(
        val_dp, embeddings="emb", targets="target", pred_probs="probs"
    )

    test_dp = problem.merge(split="test")
    test_dp = test_dp.merge(emb_dp["id", "emb"], on="id", how="left")
    result = mk.DataPanel({"id": test_dp["id"]})
    result["slice_preds"] = slicer.predict(
        test_dp, embeddings="emb", targets="target", pred_probs="probs"
    )
    result["slice_probs"] = slicer.predict_proba(
        test_dp, embeddings="emb", targets="target", pred_probs="probs"
    )

    solution = SliceDiscoverySolution(
        artifacts={
            "pred_slices": result,
        },
        attributes={
            "problem_id": problem.id,
            "slicer_class": slicer_class,
            "slicer_config": slicer_config,
            "embedding_column": "emb",
        }
    ) 
    metrics = compute_solution_metrics(
        solution,
    )
    return solution, metrics 
