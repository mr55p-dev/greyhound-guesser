import torch 
from abc import abstractmethod

class Metric:
    def __init__(self, name: str):
        self._name = name

    def get_name(self):
        return self._name

    @abstractmethod
    def log(self, pred, Y):
        ...

    @abstractmethod
    def reset(self) -> int | float:
        ...

class Accuracy(Metric):
    def __init__(self, N: int, name: str = "") -> None:
        self._N = N
        self._count = 0
        if not name:
            name = "accuracy"

        super().__init__(name)

    def log(self, pred, Y):
        is_correct = torch.eq(
            torch.argmax(pred, dim=1),
            torch.argmin(Y, dim=1),
        )
        self._count += torch.count_nonzero(is_correct).item()

    def __repr__(self):
        return f"<Accuracy metric name={self._name}>"

    def reset(self):
        ans = self._count / self._N
        self.__init__(self._N, name=self._name)
        return ans
