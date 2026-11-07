"""Keep single-subtask training separate from the full-sequence loose bank."""

from .workcell import default_bank_directory


def default_training_bank_directory():
    physical = default_bank_directory()
    return physical.with_name(physical.name + "_subtask_256")
