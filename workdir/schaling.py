import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# 32
# 444.69
# 415.15
# 402.53
# 405.72
# 420.47
# 425.13
# 446.99
# 448.70
# 405.69
# 375.80
# 426.25
# 389.71
# 404.44
# 520.50
# 445.21
# 410.22

# 16
# 490.37
# 421.47
# 497.52
# 574.25
# 477.87
# 502.99
# 429.28
# 446.25
# 426.73
# 426.18
# 480.23
# 418.86
# 439.74
# 536.11
# 483.00
# 475.06

# 8
# 630.92
# 724.47
# 496.42
# 652.70
# 654.63
# 590.34
# 570.48
# 604.33
# 879.03
# 648.20
# 545.97
# 626.55
# 585.83
# 658.09
# 576.17
# 642.37

# 4
# 678.16
# 800.19
# 757.76
# 741.31
# 718.66
# 753.82
# 721.18
# 701.13
# 903.27
# 695.89
# 799.82
# 717.87
# 670.29
# 615.98
# 815.84
# 793.81

# 2
#  992.51
# 1056.98
# 1018.06
#  968.30
#  943.55
# 1060.96
#  980.67
# 1017.79
# 1000.01
#  961.06
#  948.96
# 1026.54
# 1057.18
#  968.96
# 1107.96
# 1087.13

x_32 = np.array(
    [
        444.69,
        415.15,
        402.53,
        405.72,
        420.47,
        425.13,
        446.99,
        448.70,
        405.69,
        375.80,
        426.25,
        389.71,
        404.44,
        520.50,
        445.21,
        410.22,
    ]
)
x_16 = np.array(
    [
        490.37,
        421.47,
        497.52,
        574.25,
        477.87,
        502.99,
        429.28,
        446.25,
        426.73,
        426.18,
        480.23,
        418.86,
        439.74,
        536.11,
        483.00,
        475.06,
    ]
)
x_8 = np.array(
    [
        630.92,
        724.47,
        496.42,
        652.70,
        654.63,
        590.34,
        570.48,
        604.33,
        879.03,
        648.20,
        545.97,
        626.55,
        585.83,
        658.09,
        576.17,
        642.37,
    ]
)
x_4 = np.array(
    [
        678.16,
        800.19,
        757.76,
        741.31,
        718.66,
        753.82,
        721.18,
        701.13,
        903.27,
        695.89,
        799.82,
        717.87,
        670.29,
        615.98,
        815.84,
        793.81,
    ]
)
x_2 = np.array(
    [
        992.51,
        1056.98,
        1018.06,
        968.30,
        943.55,
        1060.96,
        980.67,
        1017.79,
        1000.01,
        961.06,
        948.96,
        1026.54,
        1057.18,
        968.96,
        1107.96,
        1087.13,
    ]
)

n_steps = 500


arr_0 = np.array([x_32, x_16, x_8, x_4, x_2])

base_line_idx = 0

cores = np.array([32, 16, 8, 4, 2])

sort = np.argsort(cores)
arr_0 = arr_0[sort, :]
cores = cores[sort]


time = np.array(arr_0) / n_steps
time_avg = np.nanmean(time, axis=1)
time_std = np.nanstd(time, axis=1)


speedup = time_avg[base_line_idx] / time_avg
scaling = speedup / (cores / cores[base_line_idx])


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


ax1.set_xlim([cores[0], cores[-1]])

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


# plt.axvspan(0, 12, alpha=0.2)

plt.savefig("scaling.png")


print(f"{cores=}")
print(f"{time_avg=}")
print(f"{speedup=}")
print(f"scaling={scaling}")
