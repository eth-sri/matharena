"""Optional monitoring shared by all container harnesses."""

from collections.abc import Mapping

from matharena.monitoring.attempt import AttemptMonitor
from matharena.monitoring.resources import ContainerResources


def monitoring_flags(value) -> dict[str, bool]:
    flags = dict.fromkeys(("diff", "cpu", "ram", "commands"), False)
    if value is None:
        return flags
    if not isinstance(value, Mapping):
        raise TypeError("harness_config.monitoring must be a mapping or null")
    unknown = value.keys() - flags.keys()
    if unknown:
        raise ValueError(f"Unknown monitoring flags: {', '.join(sorted(unknown))}")
    for key, enabled in value.items():
        if not isinstance(enabled, bool):
            raise TypeError(f"harness_config.monitoring.{key} must be a boolean")
        flags[key] = enabled
    return flags
