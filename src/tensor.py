import numpy as np
from typing import Union, Callable
import enum
import math


class _OperatorType(str, enum.Enum):
    ADD = "+"
    SUB = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    EXP = "e"
    UNIT = "Unit"
    POW = "POW"
    RELU = "ReLU"
    TAN_H = "tanh"


class Operator:
    def __init__(self, op_type: _OperatorType):
        self.op_type = op_type

    def __call__(self, *args):
        raise NotImplementedError


class Add(Operator):
    def __init__(self) -> None:
        super().__init__(op_type=_OperatorType.ADD)

    def __call__(self, *args):
        a, b = args
        a: Tensor = a if isinstance(a, Tensor) else Tensor(a)
        b: Tensor = b if isinstance(b, Tensor) else Tensor(b)

        out: Tensor = Tensor(
            data=a.data + b.data,
            _children=[a, b],
            _op=self.op_type,
        )

        def backward(reset_grad: bool = True):
            if reset_grad:
                a.zero_grad()
                b.zero_grad()
            a._grad += out._grad
            b._grad += out._grad

        out._backward = backward

        return out


class Multiply(Operator):
    def __init__(self) -> None:
        super().__init__(op_type=_OperatorType.MULTIPLY)

    def __call__(self, *args):
        a, b = args
        a: Tensor = a if isinstance(a, Tensor) else Tensor(a)
        b: Tensor = b if isinstance(b, Tensor) else Tensor(b)

        out: Tensor = Tensor(
            data=a.data * b.data,
            _children=[a, b],
            _op=self.op_type,
        )

        def backward(reset_grad: bool = True):
            if reset_grad:
                a.zero_grad()
                b.zero_grad()
            a._grad += b.data * out._grad
            b._grad += a.data * out._grad

        out._backward = backward

        return out


class Power(Operator):
    def __init__(self) -> None:
        super().__init__(op_type=_OperatorType.POW)

    def __call__(self, *args):
        a, b = args
        a: Tensor = a if isinstance(a, Tensor) else Tensor(a)

        out: Tensor = Tensor(
            data=a.data**b,
            _children=[a],
            _op=self.op_type,
        )

        def backward(reset_grad: bool = True):
            if reset_grad:
                a.zero_grad()
            a._grad += (b * a.data ** (b - 1)) * out._grad

        out._backward = backward

        return out


class ReLU(Operator):
    def __init__(self) -> None:
        super().__init__(op_type=_OperatorType.RELU)

    def __call__(self, *args):
        a, *_ = args
        a: Tensor = a if isinstance(a, Tensor) else Tensor(a)

        out: Tensor = Tensor(
            data=a.data if a.data > 0 else 0,
            _children=[a],
            _op=self.op_type,
        )

        def backward(reset_grad: bool = True):
            if reset_grad:
                a.zero_grad()
            a._grad += (out.data > 0) * out._grad

        out._backward = backward

        return out


class TanH(Operator):
    def __init__(self) -> None:
        super().__init__(op_type=_OperatorType.TAN_H)

    def __call__(self, *args):
        a, *_ = args
        a: Tensor = a if isinstance(a, Tensor) else Tensor(a)

        def tanh(x):
            return (math.exp(2 * x) - 1) / ((math.exp(2 * x) + 1))

        out: Tensor = Tensor(
            data=tanh(a.data),
            _children=[a],
            _op=self.op_type,
        )

        def backward(reset_grad: bool = True):
            if reset_grad:
                a.zero_grad()
            a._grad += (1 - (out.data**2)) * out._grad

        out._backward = backward

        return out


class Tensor:
    def __init__(
        self,
        data: float | int,
        _op: _OperatorType = _OperatorType.UNIT,
        _label: str = "unnamed_value",
        _grad: float = 0.0,
        _children: None | list["Tensor"] = None,
        _backward: Callable[[bool], None] = lambda reset_grad: None,
    ) -> None:
        self.data = data
        self._op = _op
        self._label = _label
        self._grad = _grad
        self._children = _children if _children is not None else set()
        self._backward = _backward

    def zero_grad(self):
        self._grad = 0.0

    def __add__(self, other: Union["Tensor", float, int]):
        return Add()(self, other)

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: Union["Tensor", float, int]):
        return Add()(self, -other)

    def __rsub__(self, other: Union["Tensor", float, int]):
        return -self + other

    def __mul__(self, other: Union["Tensor", float, int]):
        return Multiply()(self, other)

    def __rmul__(self, other: Union["Tensor", float, int]):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        return Power()(self, other)

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def relu(self):
        return ReLU()(self)

    def backpropagate(self, set_grad: bool = True):
        def _build_reverse_dag(root: "Tensor"):
            dag = []
            visited = set()

            def dfs(node: "Tensor"):
                if node not in visited:
                    visited.add(node)
                    for child in node._children:
                        dfs(child)
                    dag.append(node)

            dfs(root)
            return reversed(dag)

        if set_grad:
            self._grad = 1.0
        reverse_dag = _build_reverse_dag(self)

        for node in reverse_dag:
            node._backward(reset_grad=set_grad)

    def print_value(self, level: int = 0):
        tab = "\t"
        values = [f"{level * tab}{self._label} = {self.data} Op({self._op})"]
        if self._children is not None:
            for child in self._children:
                values.append(child.print_value(level=level + 1))
        return "\n".join(values)

    def __repr__(self) -> str:
        return self.print_value()
