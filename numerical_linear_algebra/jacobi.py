import numpy as np
import time


def create_tridiagonal_matrix(n):
    """创建三对角矩阵 [4,1;1,4,1;1,4,1;...,1,4,1;1,4]"""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4
        if i > 0:
            A[i, i - 1] = 1
        if i < n - 1:
            A[i, i + 1] = 1
    return A


def jacobi_rotation(A, p, q):
    """
    Jacobi旋转，消除A[p,q]和A[q,p]元素
    返回旋转矩阵参数c和s
    """
    if abs(A[p, q]) < 1e-15:
        return 1.0, 0.0

    if A[p, p] == A[q, q]:
        t = 1.0 if A[p, q] > 0 else -1.0
    else:
        theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
        t = np.sign(theta) / (abs(theta) + np.sqrt(theta**2 + 1))

    c = 1.0 / np.sqrt(1 + t**2)
    s = c * t

    return c, s


def apply_jacobi_rotation(A, V, p, q, c, s):
    """应用Jacobi旋转到矩阵A和特征向量矩阵V"""
    n = A.shape[0]

    # 更新矩阵A的第p行和第q行
    for j in range(n):
        if j != p and j != q:
            temp_pj = c * A[p, j] - s * A[q, j]
            temp_qj = s * A[p, j] + c * A[q, j]
            A[p, j] = temp_pj
            A[q, j] = temp_qj
            A[j, p] = temp_pj  # 对称矩阵
            A[j, q] = temp_qj

    # 更新对角元素
    temp_pp = c**2 * A[p, p] + s**2 * A[q, q] - 2 * s * c * A[p, q]
    temp_qq = s**2 * A[p, p] + c**2 * A[q, q] + 2 * s * c * A[p, q]

    A[p, p] = temp_pp
    A[q, q] = temp_qq
    A[p, q] = 0.0
    A[q, p] = 0.0

    # 更新特征向量矩阵V，我们直接提取列
    for i in range(n):
        temp_ip = c * V[i, p] - s * V[i, q]
        temp_iq = s * V[i, p] + c * V[i, q]
        V[i, p] = temp_ip
        V[i, q] = temp_iq


def jacobi_eigenvalue(A, max_iter=1000, tol=1e-10):
    """
    循环Jacobi方法求解特征值和特征向量
    按照(1,2),(1,3)...(2,3)(2,n)...(n-1,n)扫描
    """
    n = A.shape[0]
    A = A.copy().astype(np.float64)

    V = np.eye(n)

    for iteration in range(max_iter):
        # 检查收敛性：非对角元素的平方和
        off_diag_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag_sum += A[i, j] ** 2

        if off_diag_sum < tol:
            break

        # 按照指定顺序扫描所有(p,q)对
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) > 1e-15:
                    c, s = jacobi_rotation(A, p, q)
                    apply_jacobi_rotation(A, V, p, q, c, s)

    # 提取特征值（对角元素）
    eigenvalues = np.diag(A)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = V[:, idx]

    return eigenvalues, eigenvectors, iteration


def test_matrix_size(n):
    """测试单个矩阵大小"""
    A = create_tridiagonal_matrix(n)

    # 使用Jacobi方法求解
    start_time = time.time()
    jacobi_eigenvals, jacobi_eigenvecs, iterations = jacobi_eigenvalue(A)
    jacobi_time = time.time() - start_time

    # 使用numpy求解
    start_time = time.time()
    numpy_eigenvals, numpy_eigenvecs = np.linalg.eigh(A)
    numpy_time = time.time() - start_time

    eigenval_error = np.max(np.abs(jacobi_eigenvals - numpy_eigenvals))

    # 特征向量误差（考虑符号差异）
    eigenvec_error = 0.0
    for i in range(n):
        error1 = np.linalg.norm(jacobi_eigenvecs[:, i] - numpy_eigenvecs[:, i])
        error2 = np.linalg.norm(jacobi_eigenvecs[:, i] + numpy_eigenvecs[:, i])
        eigenvec_error = max(eigenvec_error, min(error1, error2))

    # 验证残差
    residual = np.max(
        np.abs(A @ jacobi_eigenvecs - jacobi_eigenvecs @ np.diag(jacobi_eigenvals))
    )

    return {
        "eigenvals": jacobi_eigenvals,
        "eigenvecs": jacobi_eigenvecs,
        "jacobi_time": jacobi_time,
        "numpy_time": numpy_time,
        "eigenval_error": eigenval_error,
        "eigenvec_error": eigenvec_error,
        "residual": residual,
        "iterations": iterations,
    }


def main():
    print("三对角矩阵特征值求解结果 (Jacobi vs NumPy)")
    print("=" * 80)

    for n in range(50, 101, 1):
        print(f"\n矩阵大小: {n}x{n}")
        print("-" * 40)

        result = test_matrix_size(n)

        # 紧凑输出性能比较
        print(
            f"时间(秒): Jacobi={result['jacobi_time']:.4f}, NumPy={result['numpy_time']:.4f}, 收敛迭代={result['iterations']}"
        )
        print(
            f"误差: 特征值={result['eigenval_error']:.2e}, 特征向量={result['eigenvec_error']:.2e}, 残差={result['residual']:.2e}"
        )

        print("特征值:")
        eigenvals = result["eigenvals"]
        for i in range(0, len(eigenvals), 10):
            end_idx = min(i + 10, len(eigenvals))
            vals_str = " ".join([f"{val:8.4f}" for val in eigenvals[i:end_idx]])
            print(f"  [{i+1:2d}-{end_idx:2d}] {vals_str}")

        print("特征向量(前3个,显示前5分量):")
        for i in range(min(3, n)):
            vec_str = " ".join([f"{val:7.4f}" for val in result["eigenvecs"][:5, i]])
            print(f"  v{i+1}: [{vec_str} ...]")


if __name__ == "__main__":
    main()
