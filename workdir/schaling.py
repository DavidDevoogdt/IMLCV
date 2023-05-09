from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

arr = np.array(
    [
        [1, 30, 15186.28, 44887.61],
        [2, 30, 7481.54, 22925.38],
        [3, 30, 7546.15, 23747.14],
        [4, 30, 6551.03, 17385.17],
        [5, 30, 4767.30, 13965.54],
        [6, 30, 4075.39, 12041.02],
        [7, 30, 2678.46, 8274.61],
        [8, 30, 2775.62, 8619.25],
        [9, 30, 3032.58, 9247.56],
        [10, 30, 2599.75, 7398.67],
        [12, 30, 2604.43, 8057.92],
        [16, 20, 1.921030e03, 3724.72],
        [25, 30, 1281.49, 3948.17],
        [32, 30, 1204.64, 3619.80],
        [48, 30, 1000.00, 2832.09],
        [64, 30, 915.22, 2606.41],
    ],
).T


cores = arr[0, :]
steps = arr[1, :]
time_2 = arr[2, :]
time_e = arr[3, :]

# todo: 4

time_single = (time_e - time_2) / (steps - 10)
speedup = time_single[3] / time_single
scaling = speedup / (cores / 4)

c1 = "b"
c2 = "r"
c3 = "g"

# make ax
fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)

ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax2.spines.right.set_position(("axes", 1.2))

# ax 1
(p1,) = ax1.plot(cores, scaling, "b--x")
ax1.set_xscale("log")
ax1.set_xticks(cores)

ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

ax1.set_xlabel("cores")
ax1.set_ylabel("Fraction ideal scaling", color=c1)
ax1.set_ylim([0, 1.5])

ax1.axhline(y=1.0, color="b", linestyle=":")

ax1.tick_params(axis="y", labelcolor=p1.get_color())
# ax 2

(p2,) = ax2.plot(
    cores,
    speedup,
    "r--x",
)
ax2.set_ylabel("speedup factor", color=c2)
ax2.axhline(y=1.0, color="r", linestyle=":")
ax2.tick_params(axis="y", labelcolor=p2.get_color())
ax2.set_ylim([0, 8])
# ax 3
(p3,) = ax3.plot(
    cores,
    time_single,
    "g--x",
)
ax3.set_ylabel("Time per step [s]", color=c3)

ax3.tick_params(axis="y", labelcolor=p3.get_color())
ax3.set_ylim([0, 1500])


plt.axvspan(0, 4, alpha=0.2)

plt.savefig("scaling.png")


for i in zip(cores, time_single, speedup, scaling):
    print(i)
