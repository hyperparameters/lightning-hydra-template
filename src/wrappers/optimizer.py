from typing import Any
from torch import optim


class OptimizerWrapper:
    """Wrapper for torch optimizers"""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    def __call__(self, **kwargs: Any) -> optim.Optimizer:
        """pass additional dynamic arguments to call"""

        Optimizer = getattr(optim, self.name, None)

        if Optimizer is None:
            raise ImportError(
                f"cannot import {self.name} from '{optim.__name__}' ({optim.__file__})"
            )

        kwargs.update(self.kwargs)
        return Optimizer(**kwargs)
