from .config import ExperimentConfig, load_experiment_config

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "run_experiment",
]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
