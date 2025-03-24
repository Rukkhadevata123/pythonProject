import numpy as np
import lu_decomposition


def create_special_matrix(__n=84):
    __A = np.zeros((__n, __n))
    np.fill_diagonal(__A, 6)
    np.fill_diagonal(__A[:, 1:], 1)
    np.fill_diagonal(__A[1:, :], 8)

    return __A


def create_special_b(__n=84):
    __b = np.full(__n, 15)
    __b[0] = 7
    __b[-1] = 14
    return __b


n = 84
A = create_special_matrix(n)
b = create_special_b(n)
x_exp = np.ones(n)  # 预期解全为1

# 使用三种方法求解
print("普通LU分解")
x1 = lu_decomposition.solve_linear_system(A, b)
error1 = np.linalg.norm(x1 - x_exp)
print(f"误差 ||x - x_expected|| = {error1}")
print(f"残差 ||Ax - b|| = {np.linalg.norm(A @ x1 - b)}")

print("\n列主元LU分解")
x2 = lu_decomposition.solve_with_partial_pivoting(A, b)
error2 = np.linalg.norm(x2 - x_exp)
print(f"误差 ||x - x_expected|| = {error2}")
print(f"残差 ||Ax - b|| = {np.linalg.norm(A @ x2 - b)}")

print("\n全主元LU分解")
x3 = lu_decomposition.solve_with_complete_pivoting(A, b)
error3 = np.linalg.norm(x3 - x_exp)
print(f"误差 ||x - x_expected|| = {error3}")
print(f"残差 ||Ax - b|| = {np.linalg.norm(A @ x3 - b)}")

print("\n解的所有元素")
print("预期解:", x_exp[:])
print("普通LU:", x1[:])
print("列主元:", x2[:])
print("全主元:", x3[:])
