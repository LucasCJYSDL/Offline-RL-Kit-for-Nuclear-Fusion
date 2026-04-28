"""Data preparation utilities for the fusion RL kit."""

from . import state_actuator_spaces as _sas

AVAILABLE_TASKS = _sas.AVAILABLE_TASKS
TASK_SPECS = _sas.TASK_SPECS
CURRENT_TASK = _sas.CURRENT_TASK
TASK_ARG_HELP = (
    "Task name. Canonical options: temp, rotation, dens, pres, q, betan. "
    "Legacy aliases like pres_EFIT01, q_EFIT01, and betan_EFIT01 are also accepted."
)
TASK_ALIASES = {
    "temp_EFIT01": "temp",
    "rotation_EFIT01": "rotation",
    "dens_EFIT01": "dens",
    "pres_EFIT01": "pres",
    "q_EFIT01": "q",
    "betan_EFIT01": "betan",
}


def configure_task(task_name=None):
    """Configure the active task registry and mirror the current task here."""
    spec = _sas.configure_task(task_name)
    global CURRENT_TASK
    CURRENT_TASK = _sas.CURRENT_TASK
    return spec


def resolve_task_name(task_name):
    """Map legacy task names to the canonical registry task name."""
    if task_name is None:
        return _sas.CURRENT_TASK
    task_key = task_name.lower()
    return TASK_ALIASES.get(task_key, _sas.normalize_task_name(task_key))


def get_current_task():
    """Return the active task name."""
    return _sas.CURRENT_TASK


__all__ = [
    "AVAILABLE_TASKS",
    "CURRENT_TASK",
    "TASK_ARG_HELP",
    "TASK_ALIASES",
    "TASK_SPECS",
    "configure_task",
    "get_current_task",
    "resolve_task_name",
]
