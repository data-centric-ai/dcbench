import dcbench
from domino import SpotlightSlicer, embed, DominoSlicer

from dcbench.tasks.slice_discovery.run import run_sdms

if __name__ == "__main__":

    task = dcbench.tasks["slice_discovery"]

    # solutions, metrics = run_sdms(
    #     list(task.problems.values()),
    #     slicer_class=DominoSlicer,
    #     slicer_config=dict(
    #         y_hat_log_likelihood_weight=10,
    #         y_log_likelihood_weight=10,
    #     ),
    #     encoder="clip",
    #     num_workers=10
    # )

    solutions, metrics = run_sdms(
        list(task.problems.values()),
        slicer_class=SpotlightSlicer,
        slicer_config=dict(
            # y_hat_log_likelihood_weight=10,
            # y_log_likelihood_weight=10,
            n_steps=100,
        ),
        encoder="clip",
        num_workers=0
    )
