import logging
from ortools.sat.python import cp_model

from src.autotune.parameter_space import ParameterSpace
from src.autotune import tune

from .cp_sat_parameters import CPSAT_PARAMETER_SUGGESTIONS, CPSAT_PARAMETERS
from .cp_sat_solver import CpSatSolverFactory

logger = logging.getLogger(__name__)


def tune_time_to_optimal(
        model: cp_model.CpModel,
        max_time_in_seconds: float,
        relative_gap_limit: float = 0.0,
        n_samples_for_trial: int = 10,
        n_samples_for_verification: int = 30,
        n_trials: int = 100,
        parameters: list[str] = CPSAT_PARAMETER_SUGGESTIONS
) -> dict:
    """
    Tune CP-SAT hyperparameters to minimize the time required to find an optimal solution.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned for.
        max_time_in_seconds (float): The maximum time allowed for each solve operation. Set this argument
                                    to a value sufficient for the default parameters to find an optimal solution,
                                    but not much higher as it heavily influences the runtime of the tuning process.
        relative_gap_limit (float): The relative optimality gap for considering a solution as optimal.
                                    A value of 0.0 requires the solution to be exactly optimal. Often a value of
                                    0.01 or 0.001 is used to allow for small gaps, as closing the gap to 0 can often
                                    take much longer. Defaults to 0.0.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.
        parameters (list[str]): A list of parameter names to consider for tuning. If None, all parameters will be considered. By default, a predefined list of suggested parameters is used.

    Returns:
        dict: The best parameters found during the tuning process.
    """
    logger.info("Starting tuning to minimize time to optimal solution.")

    parameter_space = ParameterSpace(
        all_parameters=CPSAT_PARAMETERS,
        parameters=parameters
    )
    parameter_space.drop_parameter("use_lns_only")  # Not useful for this metric
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])

    if relative_gap_limit > 0.0:
        parameter_space.drop_parameter("relative_gap_tolerance")

    result_params = tune.tune_time_to_optimal(
        model=model,
        solver_factory=CpSatSolverFactory(),
        max_time_in_seconds=max_time_in_seconds,
        relative_gap_limit=relative_gap_limit,
        n_samples_for_trial=n_samples_for_trial,
        n_samples_for_verification=n_samples_for_verification,
        n_trials=n_trials,
    ).params

    logger.info("Tuning for time to optimal completed.")
    return result_params


def tune_for_quality_within_timelimit(
        model: cp_model.CpModel,
        max_time_in_seconds: float,
        obj_for_timeout: int,
        direction: str,
        n_samples_for_trial: int = 10,
        n_samples_for_verification: int = 30,
        n_trials: int = 100,
        parameters: list[str] = CPSAT_PARAMETER_SUGGESTIONS
) -> dict:
    """
    Tune CP-SAT hyperparameters to maximize or minimize solution quality within a given time limit.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned.
        max_time_in_seconds (float): The time limit for each solve operation in seconds. This is the
                                    time you give the solver to find a good solution. This function
                                    is useless if you set this value too high, as it should be less
                                    than the time the solver needs to find the optimal solution with
                                    the default parameters.
        obj_for_timeout (int): The objective value to return if the solver times out.
                               This should be worse than a trivial solution.
        direction (str): A string specifying whether to 'maximize' or 'minimize' the objective value.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.
        parameters (list[str]): A list of parameter names to consider for tuning. If None, all parameters will be considered. By default, a predefined list of suggested parameters is used.

    Returns:
        dict: The best parameters found during the tuning process.

    Raises:
        ValueError: If the `direction` argument is not 'maximize' or 'minimize'.
    """
    logger.info(
        "Starting tuning for quality within time limit. Direction: %s", direction
    )

    parameter_space = ParameterSpace(
        all_parameters=CPSAT_PARAMETERS,
        parameters=parameters
    )
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])

    result_params = tune.tune_for_quality_within_timelimit(
        model=model,
        solver_factory=CpSatSolverFactory(),
        parameter_space=parameter_space,
        max_time_in_seconds=max_time_in_seconds,
        obj_for_timeout=obj_for_timeout,
        direction=direction,
        n_samples_for_trial=n_samples_for_trial,
        n_samples_for_verification=n_samples_for_verification,
        n_trials=n_trials,
    ).params

    logger.info("Tuning for quality within time limit completed.")
    return result_params


def tune_for_gap_within_timelimit(
        model: cp_model.CpModel,
        max_time_in_seconds: float,
        n_samples_for_trial: int = 10,
        n_samples_for_verification: int = 30,
        n_trials: int = 100,
        limit: float = 10,
        parameters: list[str] = CPSAT_PARAMETER_SUGGESTIONS
) -> dict:
    """
    Tune CP-SAT hyperparameters to minimize the gap within a given time limit. This is a good
    option for more complex models for which you have no chance of finding the optimal solution
    within the time limit, but you still want to have some guarantee on the quality of the solution.
    This can be considered as a proxy for the time to optimal solution.

    CAVEAT: If the time limit is too small, it will probably only minimize the presolve time, which
    can have negative effects on the long-term performance of the solver.

    Args:
        model (cp_model.CpModel): The CP-SAT model to be tuned.
        max_time_in_seconds (float): The time limit for each solve operation in seconds. It should be set
        to value where the solver with default parameters is able to find a first reasonable but not optimal
        solution. You can also try to set it to lower values.
        n_samples_for_trial (int): The number of samples to take in each trial. Defaults to 10.
        n_samples_for_verification (int): The number of samples for verifying parameters. Defaults to 30.
        n_trials (int): The number of trials to execute in the tuning process. Defaults to 100.
        limit (float): The limit for the gap. Defaults to 10. 10 should be a reasonable value for most cases,
        but if the solver with default parameters is not able to find a solution with that gap within the
        time limit, you should increase it.
        parameters (list[str]): A list of parameter names to consider for tuning. If None, all parameters will be considered. By default, a predefined list of suggested parameters is used.
    """
    logger.info("Starting tuning for gap within time limit. Limit: %s", limit)

    parameter_space = ParameterSpace(
        all_parameters=CPSAT_PARAMETERS,
        parameters=parameters
    )
    parameter_space.drop_parameter("max_time_in_seconds")
    parameter_space.filter_applicable_parameters([model])

    return tune.tune_for_gap_within_timelimit(
        model=model,
        solver_factory=CpSatSolverFactory(),
        parameter_space=parameter_space,
        max_time_in_seconds=max_time_in_seconds,
        n_samples_for_trial=n_samples_for_trial,
        n_samples_for_verification=n_samples_for_verification,
        n_trials=n_trials,
        limit=limit,
    ).params
