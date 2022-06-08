

from dataclasses import dataclass
import datetime
import uuid

from dcbench.common.table import RowMixin

from typing import List 
import os
import yaml
from .solution import Solution
from dcbench.config import config


@dataclass
class SolutionSet(RowMixin):

    set_id: str
    name: str
    summary: str

    task_id: str 

    def __post_init__(self):
        super().__init__(id=self.set_id)
    
    @property
    def local_dir(self):
        return os.path.join(config.local_dir, self.set_id)
    
    @property 
    def dir(self):
        return os.path.join(self.task_id, "solution_sets", self.set_id)

    def _write_solutions(self, containers: List[Solution]):
        ids = []
        for container in containers:
            assert isinstance(container, self.solution_class)
            ids.append(container.id)

        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate container ids in the containers passed to `write_solutions`."
            )


        path = os.path.join(self.local_dir, "solutions.yaml")
        os.makedirs(self.local_dir, exist_ok=True)
        yaml.dump(containers, open(path, "w"))
        return path

    def _write_state(self):
        path = os.path.join(self.local_dir, "state.yaml")
        os.makedirs(self.local_dir, exist_ok=True)
        yaml.dump({
            "set_id": self.set_id,
            "name": self.name,
            "summary": self.summary,
            "task_id": self.task_id,
            "solution_class": self.solution_class
        }, open(path, "w"))
        return path

    @property
    def solutions_path(self):
        return os.path.join(self.dir, "solutions.yaml")
    
    @property 
    def solution_class(self):
        import dcbench
        return dcbench.tasks[self.task_id].solution_class
        
    
    @classmethod
    def from_solutions(
        cls, 
        solutions: List[Solution], 
        name: str = None,
        summary: str = None,
        task_id: str = None,
    ):

        if len(solutions) == 0:
            raise ValueError("At least one solution must be provided.")

        if name is None:
            name = f"{datetime.date.today():%y-%m-%d-%H-%M-%S}"
        set_id = f"{name}-{str(uuid.uuid4())[:8]}"

        task_id = solutions[0].task_id
        # check that all solutions have same task id
        for solution in solutions:
            if solution.task_id != task_id:
                raise ValueError(
                    "All solutions must be from the same task."
                )

        instance = cls(
            set_id=set_id,
            name=name,
            summary=summary,
            task_id=task_id, 
        )

        instance._write_solutions(solutions)
        instance._write_state()
        return instance 

    @classmethod
    def from_dir(cls, dir: str):
        state = yaml.load(open(os.path.join(dir, "state.yaml")))
        return cls(**state)