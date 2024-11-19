import random

import numba

from typing import Iterable, Callable

import time

import minitorch

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
SimpleBackend = minitorch.TensorBackend(minitorch.SimpleOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)

LAST_TIME = None  # Variable to store the timestamp of the previous call


def default_log_fn(epoch: int, total_loss: float, correct: int, losses: Iterable[float]) -> None:
    """Default logging function for training

    Args:
    ----
        epoch : int
            The current epoch number
        total_loss : float
            The total loss for the epoch
        correct : int
            The number of correct predictions
        losses : List[float]
            The list of losses for each epoch

    Returns:
    -------
        None

    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

# class TimerLogger:
#     def __init__(self):
#         self.last_time = time.time()  # Initialize as None
#         self.epoch = 0

#     def log(self, epoch: int, total_loss: float, correct: int, losses: Iterable[float]) -> None:
#         """Logging function for timing the training.

#         Args:
#         ----
#             epoch : int
#                 The current epoch number
#             total_loss : float
#                 The total loss for the epoch
#             correct : int
#                 The number of correct predictions
#             losses : List[float]
#                 The list of losses for each epoch

#         Returns:
#         -------
#             None

#         """
#         elapsed_time = time.time() - self.last_time
#         if epoch == 0:
#             elapsed_time = 0.0
#         else:
#             elapsed_time = elapsed_time/(epoch - self.epoch)
#         self.epoch = epoch
#         print(
#             f"Epoch {epoch} | Loss: {total_loss:.4f} | Correct: {correct} | "
#             f"Average time per epoch: {elapsed_time:.4f}s"
#         )
#         self.last_time = time.time()  # Update last_time


def RParam(*shape: int, backend: minitorch.TensorBackend) -> minitorch.Parameter:
    """Creates random parameters between -1 and 1 in the desired shape with the proper backend"""
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden: int, backend: minitorch.TensorBackend):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass of the network, should be the same as calling network(x)"""
        # TODO: Implement for Task 3.5.
        # raise NotImplementedError("Need to implement for Task 3.5")
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x).sigmoid()
        return x


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int, backend: minitorch.TensorBackend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass of the network, should be the same as calling network(x)"""
        # TODO: Implement for Task 3.5.
        # raise NotImplementedError("Need to implement for Task 3.5")
        #now we can simply use matmul for this
        return (x @ self.weights.value) + self.bias.value


class FastTrain:
    def __init__(self, hidden_layers: int, backend: minitorch.TensorBackend = FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x: Iterable[float]) -> minitorch.Tensor:
        """Run the model on a single input"""
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X: Iterable[Iterable[float]]) -> minitorch.Tensor:
        """Run the model on a batch of inputs"""
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data: minitorch.Graph, learning_rate: int, max_epochs:int=500, log_fn:Callable[[int, float, int, list], None] = default_log_fn) -> None:
        """Function to train the model on the given data

        Args:
        ----
            data : minitorch.Graph
                The data to train on
            learning_rate : int
                The learning rate for the model
            max_epochs : int
                The maximum number of epochs to train
            log_fn : Callable[[int, float, int, list], None]
                The logging function to use

        Returns:
        -------
            None

        """
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 10
        losses = []

        for epoch in range(max_epochs):
            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)
                # Forward

                out = self.model.forward(X).view(y.shape[0])
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()
                (loss / y.shape[0]).sum().view(1).backward()

                total_loss = loss.sum().view(1)[0]

                # Update
                optim.step()

            losses.append(total_loss)
            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")
    parser.add_argument("--TIME", action="store_true", help="Enable timing functionality.")
    parser.add_argument("--MAX_EPOCHS", type=int, default=500, help="Maximum number of epochs to train.")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    #commented out because pyright really hates this and I can't get it to work without changing function signatures.
    # if args.BACKEND == "gpu":
    #     backend = GPUBackend
    # elif args.BACKEND == "cpu":
    #     backend = FastTensorBackend
    # elif args.BACKEND == "simple":
    #     backend = SimpleBackend
    # else:
    #     raise(ValueError(f"Unknown backend {args.BACKEND}"))

    # if args.TIME:
    #     timer = TimerLogger()
    #     FastTrain(
    #         HIDDEN, backend=backend
    #     ).train(data, RATE, log_fn=timer.log, max_epochs=args.MAX_EPOCHS)
    # else:
    #     FastTrain(
    #         HIDDEN, backend=backend
    #     ).train(data, RATE, log_fn=default_log_fn, max_epochs=args.MAX_EPOCHS)

    FastTrain(
        HIDDEN, backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    ).train(data, RATE, log_fn=default_log_fn, max_epochs=args.MAX_EPOCHS)
