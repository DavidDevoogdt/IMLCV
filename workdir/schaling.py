import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# arr = np.array(
#     [
#         [1, 30, 15186.28, 44887.61],
#         [2, 30, 7481.54, 22925.38],
#         [3, 30, 7546.15, 23747.14],
#         [4, 30, 6551.03, 17385.17],
#         [5, 30, 4767.30, 13965.54],
#         [6, 30, 4075.39, 12041.02],
#         [7, 30, 2678.46, 8274.61],
#         [8, 30, 2775.62, 8619.25],
#         [9, 30, 3032.58, 9247.56],
#         [10, 30, 2599.75, 7398.67],
#         [12, 30, 2604.43, 8057.92],
#         [16, 20, 1.921030e03, 3724.72],
#         [25, 30, 1281.49, 3948.17],
#         [32, 30, 1204.64, 3619.80],
#         [48, 30, 1000.00, 2832.09],
#         [64, 30, 915.22, 2606.41],
#     ],
# ).T

n_steps = 100


arr_0 = np.array(
    [
        [5901, 6285, 5878, 5535, 5988, 8643, 5608, 5239, 5191],
        [18657, 18917, 23300, 20127, 19076, np.NaN, 17745, 16322, 16198],
        [13742, 14719, 14186, 12896, 11550, 16017, 12772, 11842, 11762],
        [8787, 8625, 8538, 8031, 7319, np.NaN, 8145, 7499, 7679],
        [5235, 5348, 5275, 4931, 4999, 7442, 5087, 5718, 5570],
        [6620, 6485, 5746, 5211, np.NaN, np.NaN, 5836, np.NaN, 4988],
        [np.NaN, 8119, 7747, 7247, np.NaN, np.NaN, 8174, np.NaN, 6960],
        [np.NaN, 15319, 14701, 13641, 17650, np.NaN, 14008, np.NaN, 12834],
        [9958, 10406, 13720, 10361, np.NaN, np.NaN, 10540, np.NaN, 9506],
    ],
)

base_line_idx = 1

cores = np.array([64, 8, 16, 32, 128, 49, 36, 12, 25])

sort = np.argsort(cores)
arr_0 = arr_0[sort, :]
cores = cores[sort]


time = np.array(arr_0) / n_steps
time_avg = np.nanmean(time, axis=1)
time_std = np.nanstd(time, axis=1)


speedup = time_avg[base_line_idx] / time_avg
scaling = speedup / (cores / cores[base_line_idx])


# speedup = vmap(lambda x: time_avg[base_line_idx] / x)(time)
# scaling = vmap(lambda x, y: x / (y / cores[base_line_idx]))(speedup, cores)

# print(speedup)
# print(scaling)

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
p1 = ax1.plot(
    cores,
    scaling,
    # yerr=np.nanstd(scaling, axis=1),
    "b--x",
)
ax1.set_xscale("log")
ax1.set_xticks(cores)

ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

ax1.set_xlabel("cores")
ax1.set_ylabel("Fraction ideal scaling", color=c1)
ax1.set_ylim([0, 1.2])

ax1.axhline(y=cores[0], color="b", linestyle=":")

ax1.tick_params(axis="y", labelcolor=p1[0].get_color())
# ax 2

p2 = ax2.plot(
    cores,
    speedup,
    # yerr=np.nanstd(speedup, axis=1),
    "r--x",
)
ax2.set_ylabel("speedup factor", color=c2)
# ax2.axhline(y=1.0, color="r", linestyle=":")
ax2.tick_params(axis="y", labelcolor=p2[0].get_color())
ax2.set_ylim([0.5, np.ceil(speedup[-1])])
# ax 3
p3 = ax3.errorbar(
    x=cores,
    y=np.nanmean(time, axis=1),
    yerr=np.nanstd(time, axis=1),
    fmt="g--x",
)
ax3.set_ylabel("Time per step [s]", color=c3)

ax3.tick_params(axis="y", labelcolor=p3[0].get_color())
# ax3.set_ylim([0, time_single[0]*1.1 ])


plt.axvspan(0, 12, alpha=0.2)

plt.savefig("scaling.png")


print(f"{cores=}")
print(f"{time_avg=}")
print(f"{speedup=}")
print(f"scaling={scaling}")
