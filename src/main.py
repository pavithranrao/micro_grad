from tensor import Tensor, ReLU
from nn import MLP
import numpy as np


def main():
    # y = mx + c
    # x = Tensor(10, _label="x")
    # m = Tensor(3, _label="m")
    # c = Tensor(2, _label="c")
    # mx = m * x
    # mx._label = "mx"

    # y: Tensor = mx - c
    # y._label = "y"
    # print(y)

    # y.zero_grad()
    # y.backpropagate()

    # print("x", x._grad)
    # print("m", m._grad)
    # print("c", c._grad)

    # d = Tensor(10)
    # e = 2 / d
    # print(e)
    # e.backpropagate(set_grad=True)
    # print(d._grad)

    # print(ReLU()(Tensor(0.5)))

    mlp = MLP(
        nin=3,
        nouts=[3, 4, 1],
    )
    data = np.array(
        [
            [1.8, 3.3, 5.4, 0.9],
            [1, 6.9, 2.8, 9.8],
            [2.9, 1.9, 6.5, 7.3],
            [2.5, 7.2, 0, 8.1],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    learning_rate = 0.001

    for epoch in range(20):
        total_loss: Tensor = Tensor(0)
        for data_pt, label in zip(data, labels):
            pred = mlp(np.array(data_pt))
            print(f"pred={pred.data} {label=}")
            loss: Tensor = (pred - label) ** 2
            total_loss += loss

        total_loss.backpropagate()
        for p in mlp.parameters():
            p.data -= learning_rate * p._grad

        avg_loss = total_loss / len(data)
        print(f"{epoch=} avg_loss={avg_loss.data}")
        print("=" * 100)


if __name__ == "__main__":
    main()
