import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_function(f, a, b):
    def f(x):
        return 2.74*(x**3) - 1.93*(x**2) - 15.28*x - 3.72

    x_min, x_max = -4, 4

    xs = np.linspace(x_min, x_max, 400)
    ys = [f(xx) for xx in xs]

    plt.figure(figsize=(8,6))
    plt.title("График f(x) = 2.74x^3 - 1.93x^2 - 15.28x - 3.72;\nИнтервал x ∈ [-4, 4]")

    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, color='k', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.xlabel("x in [-4,4]")
    plt.ylabel("f(x)")

    test_points = [-3, -2, -1, 0, 1, 2, 3]
    for tp in test_points:
        val = f(tp)
        plt.plot(tp, val, 'ro')
        plt.text(tp, val, f"({tp}, {val:.1f})", fontsize=8)

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_function_with_root(f, a, b, root):
    xs = np.linspace(a, b, 400)
    ys = [f(xx) for xx in xs]

    plt.figure(figsize=(8,6))
    plt.title(f"График f(x) на [{a},{b}] с отмеченным корнем")
    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, color='k', linewidth=1)

    plt.xlim(a, b)
    min_y = min(ys)
    max_y = max(ys)
    margin = 0.1*(max_y - min_y) if max_y>min_y else 1
    plt.ylim(min_y - margin, max_y + margin)

    plt.xlabel(f"x in [{a},{b}]")
    plt.ylabel("f(x)")

    fxr = f(root)
    plt.plot(root, fxr, 'ro')
    plt.text(root, fxr, f" root=({root:.3f}, {fxr:.3f})", fontsize=9)

    plt.grid(True)
    plt.legend()
    plt.show()


def plot_system(F1, F2,
                               x_min, x_max,
                               y_min, y_max,
                               solutions):

    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(xs, ys)

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)

    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            xx = X[i,j]
            yy = Y[i,j]
            Z1[i,j] = F1(xx, yy)
            Z2[i,j] = F2(xx, yy)

    plt.figure(figsize=(8,6))
    plt.title(f"График системы F1=sin(x+1)-y=1.2; \nF2=2x+cos(y)=2; \nx∈[{x_min},{x_max}], y∈[{y_min},{y_max}]")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    c1 = plt.contour(X, Y, Z1, levels=[0], colors='blue')
    c2 = plt.contour(X, Y, Z2, levels=[0], colors='red')

    for (x_s, y_s) in solutions:
        plt.plot(x_s, y_s, 'mo')  # magenta circle
        plt.text(x_s, y_s, f"({x_s:.3f}, {y_s:.3f})", fontsize=9)

    lineF1 = mlines.Line2D([], [], color='blue', label='F1=0')
    lineF2 = mlines.Line2D([], [], color='red',  label='F2=0')
    plt.legend(handles=[lineF1, lineF2], loc='upper right')

    plt.xlabel(f"x in [{x_min},{x_max}]")
    plt.ylabel(f"y in [{y_min},{y_max}]")
    plt.grid(True)
    plt.show()
