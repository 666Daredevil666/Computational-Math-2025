import numpy as np
import random


def read_data_from_keyboard():
    print("Введите размерность матрицы n (1-20): ")
    n = int(input().strip())
    if n > 20 or n <= 0:
        raise ValueError("Недопустимое значение n.")

    A = []
    print(f"Введите матрицу A ({n}x{n} построчно, числа через пробел):")
    for _ in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            raise ValueError("Неверное количество элементов в строке.")
        A.append(row)

    print(f"Введите вектор b ({n} чисел через пробел):")
    b = list(map(float, input().split()))
    if len(b) != n:
        raise ValueError("Неверное количество элементов в векторе b.")

    return n, A, b


def read_data_from_file(filename="input.txt"):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    n = int(lines[idx])
    idx += 1
    A = []
    for _ in range(n):
        A.append(list(map(float, lines[idx].split())))
        idx += 1

    b = list(map(float, lines[idx].split()))
    return n, A, b


def generate_random_system(n):
    A = []
    for i in range(n):
        row = [random.uniform(-10, 10) for _ in range(n)]
        diag_temp = row[i]
        row[i] = 0
        sum_other = sum(abs(x) for x in row)
        sign = random.choice([-1, 1])
        row[i] = sign * (sum_other + random.uniform(1, 5))
        A.append(row)

    b = [random.uniform(-10, 10) for _ in range(n)]
    return np.array(A, dtype=float), np.array(b, dtype=float)


def check_and_make_diagonally_dominant(A, b):
    n = len(A)
    for i in range(n):
        max_row = i
        max_val = abs(A[i][i])
        for j in range(i, n):
            current = abs(A[j][i])
            if current > max_val:
                max_val = current
                max_row = j

        if max_val == 0:
            return False

        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
    return True


def simple_iterations(A, b, eps=1e-6, max_iter=1000):
    n = len(A)
    D = np.diag(A)
    if any(d == 0 for d in D):
        raise ValueError("На диагонали есть нулевые элементы после перестановки.")

    B = -np.array(A) / D[:, None]
    np.fill_diagonal(B, 0)
    c = np.array(b) / D

    x = np.zeros(n)
    for it in range(1, max_iter + 1):
        x_new = B @ x + c
        if np.linalg.norm(x_new - x, np.inf) < eps:
            return x_new, it, (x_new - x)
        x = x_new
    return x, max_iter, (x - x_new)


def print_results(A, b, x, iterations, diff, eps):
    print("\nРезультаты:")
    print(f"Количество итераций: {iterations}")
    print("Вектор решения:")
    print(np.array2string(x, precision=9, suppress_small=True))

    print("\nВектор погрешностей (последняя итерация):")
    print(np.array2string(diff, precision=2, suppress_small=True))

    residual = A @ x - b
    print("\nВектор невязки (A*x - b):")
    print(np.array2string(residual, precision=2, suppress_small=True))

    print(f"\nПроверка точности (||Δx|| < {eps}): {np.linalg.norm(diff, np.inf):.2e}")


def compare_with_numpy(A, b):
    try:
        x_lib = np.linalg.solve(A, b)
        print("\nРешение с использованием numpy.linalg.solve:")
        print(np.array2string(x_lib, precision=9, suppress_small=True))
        return x_lib
    except np.linalg.LinAlgError:
        print("\nМатрица вырождена, библиотечное решение невозможно.")
        return None


def main():
    print("Лабораторная работа: Метод простых итераций")
    print("=" * 50)

    source = input("Ввод данных:\n1 - Клавиатура\n2 - Файл\n3 - Случайная генерация\nВыбор: ").strip()

    if source == '1':
        n, A, b = read_data_from_keyboard()
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
    elif source == '2':
        n, A, b = read_data_from_file()
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
    elif source == '3':
        print("Введите размерность матрицы n (1-20): ")
        n = int(input().strip())
        if n > 20 or n <= 0:
            print("Недопустимое значение n!")
            return
        A, b = generate_random_system(n)
        print("\nСгенерированная матрица A:")
        print(np.array2string(A, precision=2, suppress_small=True))
        print("\nСгенерированный вектор b:")
        print(np.array2string(b, precision=2, suppress_small=True))
    else:
        print("Неверный ввод!")
        return

    if not check_and_make_diagonally_dominant(A, b):
        print("Невозможно достичь диагонального преобладания!")
        return

    eps = float(input("Введите точность (например, 1e-6): ").strip() or 1e-6)

    try:
        x, iterations, diff = simple_iterations(A, b, eps)
    except ValueError as e:
        print(f"Ошибка: {str(e)}")
        return

    print_results(A, b, x, iterations, diff, eps)
    x_lib = compare_with_numpy(A, b)

    if x_lib is not None:
        diff = x - x_lib
        print("\nРазница с библиотечным решением:")
        print(f"Максимальная: {np.abs(diff).max():.2e}")
        print(f"Средняя: {np.abs(diff).mean():.2e}")
        print("\nОбъяснение: Различия вызваны:")
        print("- Ограниченным числом итераций метода")
        print("- Накоплением ошибок округления")
        print("- Особенностями метода (диагональное преобладание)")


if __name__ == "__main__":
    main()