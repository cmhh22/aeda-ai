from abc import ABC, abstractmethod
from typing import Any


class DataComponent(ABC):
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
