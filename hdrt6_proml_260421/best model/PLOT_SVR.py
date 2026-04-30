import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

model = joblib.load("best_svr_model.pkl")

wind = 5
altitude = 10
rotation = 0
distance = 0

# =========================
# x-axis logic
# =========================
if distance > 0:
    x_vals = np.linspace(0, distance, 20)
    x_label = "Distance (m)"
else:
    x_vals = np.linspace(0, altitude, 20)
    x_label = "Altitude (m)"

# =========================
# input dataframe
# =========================
if distance == 0:
    df_input = pd.DataFrame({
        "풍속(m/s)": [wind] * len(x_vals),
        "비행고도(m)": x_vals,
        "2D이동거리(m)": [0] * len(x_vals),
        "총회전량(deg)": [rotation] * len(x_vals)
    })
else:
    df_input = pd.DataFrame({
        "풍속(m/s)": [wind] * len(x_vals),
        "비행고도(m)": [altitude] * len(x_vals),
        "2D이동거리(m)": x_vals,
        "총회전량(deg)": [rotation] * len(x_vals)
    })

pred = model.predict(df_input)
pred = np.clip(pred, 0, 100)
battery = 100 - pred

# =========================
# cutoff logic
# =========================
idx = np.where(battery <= 0)[0]
if len(idx) > 0:
    cut = idx[0] + 2
    x_vals = x_vals[:cut]
    battery = battery[:cut]

# =========================
# WEB STYLE PLOT
# =========================
plt.style.use("seaborn-v0_8-white")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")

# 라인
ax.plot(
    x_vals,
    battery,
    color="#2ECC71",
    linewidth=3,
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=2
)

# =========================
# 상태 구간
# =========================
ax.axhspan(25, 30, color="#F1C40F", alpha=0.12)
ax.axhspan(15, 20, color="#E67E22", alpha=0.12)
ax.axhspan(5, 10, color="#E74C3C", alpha=0.12)

# =========================
# 텍스트 UI
# =========================
left_x = x_vals[0] + (x_vals[-1] - x_vals[0]) * 0.02

ax.text(left_x, 27.5, "WARNING",
        fontsize=11, color="#B7950B", weight="bold")

ax.text(left_x, 17.5, "RTL",
        fontsize=11, color="#D35400", weight="bold")

ax.text(left_x, 7.5, "CRITICAL",
        fontsize=11, color="#C0392B", weight="bold")

# =========================
# UI polish
# =========================
ax.set_title("Battery Curve", fontsize=16, weight="bold", pad=10)
ax.set_xlabel(x_label, fontsize=12)
ax.set_ylabel("Battery (%)", fontsize=12)

y_max = min(100, max(battery) * 1.1)
ax.set_ylim(0, y_max)
ax.set_xlim(left=0)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()