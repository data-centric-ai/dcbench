import logging
import os
from typing import Optional

import click

from dcbench.common import Solution

from .constants import LOCAL_DIR
from .version import __version__

__all__ = ("main",)
_log = logging.getLogger(__name__)

# flake8: noqa
BANNER = """
       ____  _________    ____   ____  _______   __________  __ 
      / __ \/ ____/   |  /  _/  / __ )/ ____/ | / / ____/ / / /
     / / / / /   / /| |  / /   / __  / __/ /  |/ / /   / /_/ / 
    / /_/ / /___/ ___ |_/ /   / /_/ / /___/ /|  / /___/ __  /  
   /_____/\____/_/  |_/___/  /_____/_____/_/ |_/\____/_/ /_/   
                                                        
    """


class MainGroup(click.Group):
    def format_usage(self, ctx, formatter):
        formatter.write(BANNER)
        super().format_usage(ctx, formatter)


@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"], auto_envvar_prefix="DCBENCH"
    ),
    cls=MainGroup,
)
@click.version_option(prog_name="dcbench", version=__version__)
@click.option(
    "--optional-artefacts-url",
    help="The URL pointing to the optional artefacts bundle.",
)
@click.option(
    "--working-dir",
    help="Set the working directory (default: %s)."
    % os.path.join(os.getcwd(), LOCAL_DIR),
)
def main(
    hidden_artefacts_url: Optional[str] = None,
    working_dir: Optional[str] = None,
    **kwargs
):
    """Collection of benchmarks that test various aspects of ML data
    preprocessing and management."""
    global HIDDEN_ARTEFACTS_URL
    HIDDEN_ARTEFACTS_URL = hidden_artefacts_url
    global LOCAL_DIR
    if working_dir is not None:
        LOCAL_DIR = working_dir


@main.command(help="List all available scenarios.")
def scenarios():
    for id in Scenario.list():
        click.echo(id)


@main.command(
    help="List solutions for a given scenario and corresponding evaluation results if available."
)
@click.option("--scenario-id", type=str, help="The ID of the scenario.", required=True)
def solutions(scenario_id: str):
    scenario = Scenario.scenarios.get(scenario_id, None)()
    if scenario is None:
        click.echo(
            "The scenario with identifier '%s' not found." % scenario_id, err=True
        )
        return
    click.echo(scenario.solutions)


@main.command(help="Create a new solution for a given scenario.")
@click.option("--scenario-id", type=str, help="The ID of the scenario.", required=True)
@click.option("--name", type=str, help="The name of the new solution.")
@click.option(
    "--paper",
    type=str,
    help="The URL pointing to a paper describing the solution method.",
)
@click.option(
    "--code",
    type=str,
    help="The URL pointing to a repository or notebook containing the solution code.",
)
@click.option(
    "--artefacts-url", type=str, help="The URL pointing to the solution artefacts."
)
def new_solution(
    scenario_id: str,
    name: Optional[str],
    paper: Optional[str],
    code: Optional[str],
    artefacts_url: Optional[str],
):
    scenario = Scenario.scenarios.get(scenario_id, None)()
    if scenario is None:
        click.echo(
            "The scenario with identifier '%s' not found." % scenario_id, err=True
        )
        return
    solution = Solution(
        scenario, name=name, paper=paper, code=code, artefacts_url=artefacts_url
    )
    solution.save()
    click.echo("New solution saved to:", err=True)
    click.echo(solution.location)


@main.command(help="Evaluate solutions for one or more scenarios.")
@click.option(
    "--scenario-id",
    type=str,
    help="The ID of the scenario. If omitted then all scenarios are considered.",
)
@click.option(
    "--force",
    type=bool,
    is_flag=True,
    help="Evaluates even if a previous evaluation result exists.",
)
def solve(scenario_id: Optional[str], force: bool):
    scenarios = []
    if scenario_id is not None:
        scenario = Scenario.scenarios.get(scenario_id, None)
        if scenario is None:
            click.echo(
                "The scenario with identifier '%s' not found." % scenario_id, err=True
            )
            return
        scenarios.append(scenario)
    else:
        scenarios = [
            Scenario.scenarios[id]() for id in sorted(Scenario.scenarios.keys())
        ]

    for scenario in scenarios:
        for solution in scenario.solutions.values():
            if solution.result is not None or force:
                click.echo(
                    "Evaluating solution '%s' of scenario '%s'."
                    % (solution.id, scenario.id),
                    err=True,
                )
                solution.evaluate()
                solution.save()
                click.echo("Result:", err=True)
                click.echo(solution.result)


@main.command(help="Evaluate solutions for one or more scenarios.")
@click.option(
    "--scenario-id",
    type=str,
    help="The ID of the scenario. If omitted then all scenarios are considered.",
)
@click.option(
    "--force",
    type=bool,
    is_flag=True,
    help="Evaluates even if a previous evaluation result exists.",
)
def evaluate(scenario_id: Optional[str], force: bool):
    scenarios = []
    if scenario_id is not None:
        scenario = Scenario.scenarios.get(scenario_id, None)
        if scenario is None:
            click.echo(
                "The scenario with identifier '%s' not found." % scenario_id, err=True
            )
            return
        scenarios.append(scenario)
    else:
        scenarios = [
            Scenario.scenarios[id]() for id in sorted(Scenario.scenarios.keys())
        ]

    for scenario in scenarios:
        for solution in scenario.solutions.values():
            if solution.result is not None or force:
                click.echo(
                    "Evaluating solution '%s' of scenario '%s'."
                    % (solution.id, scenario.id),
                    err=True,
                )
                solution.evaluate()
                solution.save()
                click.echo("Result:", err=True)
                click.echo(solution.result)


if __name__ == "__main__":
    main(prog_name="dcbench")
