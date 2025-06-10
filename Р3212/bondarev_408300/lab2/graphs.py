import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Для некоторых сред IDE (PyCharm и т.п.)
import matplotlib.pyplot as plt

def f(x):
    return 2.74*(x**3) - 1.93*(x**2) - 15.28*x - 3.72

x_min, x_max = -4, 4

xs = np.linspace(x_min, x_max, 400)
ys = [f(xx) for xx in xs]

plt.figure(figsize=(8,6))
plt.title("График f(x) = 2.74x^3 - 1.93x^2 - 15.28x - 3.72; \n Интервал x ∈ [-4, 4]")

# Основная кривая:
plt.plot(xs, ys, label="f(x)")
# Горизонтальная ось x (ось f(x)=0):
plt.axhline(0, color='k', linewidth=1)

# Ограничение диапазонов осей:
plt.xlim(x_min, x_max)
# y-границы подберем автоматически или умножим при желании:
# plt.ylim(min(ys)*1.1, max(ys)*1.1)

# Подписи осей:
plt.xlabel("x in [-4,4]")
plt.ylabel("f(x)")

# Несколько «контрольных» точек с подписями:
test_points = [-3, -2, -1, 0, 1, 2, 3]
for tp in test_points:
    val = f(tp)
    plt.plot(tp, val, 'ro')  # 'r'=red,'o'=marker
    plt.text(tp, val, f"({tp}, {val:.1f})", fontsize=8)

plt.grid(True)
plt.legend()
plt.show()
