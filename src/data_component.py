from abc import ABC, abstractmethod
from typing import Any

from .data_component_contracts import ComponentOutput


class DataComponent(ABC):
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> ComponentOutput:
        raise NotImplementedError
