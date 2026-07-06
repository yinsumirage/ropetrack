"""Small cached-prediction refinement utilities."""

__all__ = ["RopePoseRefiner"]


def __getattr__(name: str):
    if name == "RopePoseRefiner":
        from .rope_refiner import RopePoseRefiner

        return RopePoseRefiner
    raise AttributeError(name)
