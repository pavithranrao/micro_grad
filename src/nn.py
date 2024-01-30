import random
import tensor
import itertools
from typing import Any
import random


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> list[tensor.Tensor]:
        return []


class Neuron(Module):
    def __init__(
        self,
        nin: int,
        nonlin: bool = True,
    ):
        self.w = [tensor.Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = tensor.Tensor(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # print(type(x), type(self.w), len(self.w))
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[tensor.Tensor]:
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(
        self,
        nin: int,
        nout: list[int],
        dropout: float = 0.0,
        last_layer: bool = False,
        **kwargs: dict[Any, Any],
    ):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        self.dropout = dropout
        self.last_layer = last_layer

    def __call__(self, x):
        out = []
        if self.dropout > 0.0:
            for n in self.neurons:
                if random.random() >= self.dropout:
                    out.append(n(x))
                else:
                    # print(f"Dropping out the neuron")
                    out.append(tensor.Tensor(0.0))
        else:
            out = [n(x) for n in self.neurons]
        return out[0] if self.last_layer else out

    def parameters(self) -> list[tensor.Tensor]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(
        self,
        nin: int,
        nouts: list[int],
        dropouts: float | list[float] | None = None,
    ):
        sz = [nin] + nouts
        num_layers = len(sz)
        self.layers = [
            Layer(layer_in, layer_out, nonlin=True)
            for layer_in, layer_out in itertools.pairwise(sz)
        ]
        self.layers[-1].nonlin = False
        self.layers[-1].last_layer = True
        if dropouts:
            if isinstance(dropouts, float):
                for idx in range(num_layers - 1):
                    self.layers[idx].dropout = dropouts
            elif isinstance(dropouts, list):
                if len(dropouts) != num_layers - 1:
                    raise Exception(
                        "Dropouts must be equal to number of intermediate layers",
                    )
                for idx in range(num_layers - 1):
                    self.layers[idx].dropout = dropouts[idx]
            else:
                raise Exception(
                    f"droputs can be either int or list[int], recieved {type(dropouts)}"
                )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[tensor.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
