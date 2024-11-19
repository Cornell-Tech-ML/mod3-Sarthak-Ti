import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.SimpleOps)


def run_matmul(backend: minitorch.TensorBackend, size: int=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y
    return z


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

#now create a bar plot and then save it. th eplot should be 4 paired bars, (so 8 total bars) x value is n trials, y value is time in seconds
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
bar_width = 0.35
bar_positions = np.arange(len(times))
bar_positions = bar_positions * 2 * bar_width
fast_times = [times[size]["fast"] for size in times]
gpu_times = [times[size]["gpu"] for size in times]
ax.bar(bar_positions, fast_times, bar_width, label="Fast")
ax.bar(bar_positions + bar_width, gpu_times, bar_width, label="Simple")
ax.set_xticks(bar_positions + bar_width / 2)
ax.set_xticklabels([str(size) for size in times])
ax.set_xlabel("Size")
ax.set_ylabel("Time (s)")
ax.set_title("Matrix Multiplication Timings")
ax.legend()
#save plot
plt.savefig("timing_simple.png")