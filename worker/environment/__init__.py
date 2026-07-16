"""Environment module — openmathinstruct only (OpenCode excluded)."""

from worker.environment.base import Environment
from worker.environment.openmathinstruct import OpenMathInstructEnvironment


def load_environment(name: str) -> Environment:
    if name == "openmathinstruct":
        return OpenMathInstructEnvironment()
    raise ValueError(f"Unknown environment: {name}")


def load_environments(names: list[str]) -> dict[str, Environment]:
    return {name: load_environment(name) for name in names}


__all__ = ["Environment", "load_environment", "load_environments"]
