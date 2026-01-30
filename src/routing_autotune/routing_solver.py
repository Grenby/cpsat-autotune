from ..autotune.models import SolverFactory, ModelSolver, GapModelSolver, GapIntervalModelSolver,SolutionStatus
from ortools.sat.python import cp_model

import logging
from dataclasses import dataclass
import numpy as np
from ortools.sat.python import cp_model

from cpsat_autotune.cp_sat_parameters import get_parameter_by_name
from ortools.sat import sat_parameters_pb2

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()  # You can add more handlers (e.g., file handlers) as needed
    ],
)

class RoutingSatSolver(ModelSolver[cp_model.CpModel]):
    
    def __init__(self, solver: cp_model.CpSolver):
        self.solver = solver
    
    @property
    def seed(self) -> int:
        return self.solver.parameters.random_seed
    
    @seed.setter
    def seed(self,_seed:int):
        self.solver.parameters.random_seed = _seed

    def solve(self, model : cp_model.CpModel)->SolutionStatus:
        status = self.solver.solve(model)
        if status == cp_model.FEASIBLE:
            return SolutionStatus.FEASIBLE
        if status == cp_model.OPTIMAL:
            return SolutionStatus.OPTIMAL
        if status == cp_model.INFEASIBLE:
            return SolutionStatus.INFEASIBLE
        return SolutionStatus.NOT_FOUND
    
    @property
    def best_objective(self):
        return self.best_objective
    
    
    @property
    def best_bound(self):
        return self.best_bound

    @property
    def max_time_seconds(self):
        return self.solver.parameters.max_time_in_seconds

    @max_time_seconds.setter
    def max_time_seconds(self, time_seconds : int):
        self.solver.parameters.max_time_in_seconds = time_seconds

class RoutingSolverFactory(SolverFactory[cp_model.CpModel]):
    def prepare_solver(self, params: dict[str, float | int | bool | list | tuple]) -> RoutingSatSolver:
        solver = cp_model.CpSolver()
        subsolver = sat_parameters_pb2.SatParameters()
        subsolver.name = "tuned_solver"
        has_subsolver_params = False
        for key, value in params.items():
            is_subsolver_param = get_parameter_by_name(key).subsolver
            level = subsolver if is_subsolver_param else solver.parameters
            if is_subsolver_param:
                has_subsolver_params = True
            if isinstance(value, (list, tuple)):
                getattr(level, key).extend(value)
            else:
                setattr(level, key, value)
        for key, value in self.fixed_params.items():
            is_subsolver_param = get_parameter_by_name(key).subsolver
            level = subsolver if is_subsolver_param else solver.parameters
            if is_subsolver_param:
                has_subsolver_params = True
            if isinstance(value, (list, tuple)):
                getattr(level, key).extend(value)
            else:
                setattr(level, key, value)
        if has_subsolver_params:
            solver.parameters.subsolver_params.append(subsolver)
            solver.parameters.extra_subsolvers.append(subsolver.name)
        logging.debug("Solver prepared with params: %s", params)

        return RoutingSatSolver(solver=solver)
