import statistics
import time

import numpy as np


def benchmark_list(data, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = [x * 2 for x in data]
        end = time.perf_counter()
        times.append(end - start)
    return times, out


def benchmark_numpy(data, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = data * 2
        end = time.perf_counter()
        times.append(end - start)
    return times, out


def main():
    n = 1_000_000
    repeats = 5

    py_list = list(range(n))
    np_array = np.arange(n, dtype=np.int64)

    list_times, list_out = benchmark_list(py_list, repeats=repeats)
    np_times, np_out = benchmark_numpy(np_array, repeats=repeats)

    assert list_out[123456] == int(np_out[123456])

    list_avg = statistics.mean(list_times)
    np_avg = statistics.mean(np_times)
    speedup = list_avg / np_avg if np_avg else float("inf")

    print("NumPy Speed Test")
    print(f"Data size: {n:,} numbers")
    print(f"Repeats: {repeats}")
    print()
    print("Execution Time")
    print(f"Python list average: {list_avg:.6f} s")
    print(f"NumPy array average:  {np_avg:.6f} s")
    print(f"NumPy speedup:        {speedup:.2f}x")
    print()
    print("Three Observations")
    print(
        "1. NumPy is much faster for large numeric operations because "
        "vectorized execution avoids Python-level loops."
    )
    print(
        "2. NumPy timings are typically more consistent across runs, while "
        "Python list-comprehension timings fluctuate more."
    )
    print(
        "3. With 1M values, NumPy saves substantial runtime; this gap usually "
        "grows as data size increases."
    )


if __name__ == "__main__":
    main()
