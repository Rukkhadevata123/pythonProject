import numpy as np
import time
import matplotlib.pyplot as plt
from cond import condition_number
from main4 import create_hilbert_matrix


def compare_condition_numbers():
    """比较不同阶Hilbert矩阵的条件数"""
    # 测试的Hilbert矩阵阶数范围
    n_range = range(5, 21)
    
    # 存储结果的列表
    cond_numpy = []
    cond_estimate = []
    times_numpy = []
    times_estimate = []
    
    print(f"{'Order':<5}{'NumPy Cond':<20}{'Est. Cond':<20}{'NumPy Time(s)':<15}{'Est. Time(s)':<15}")
    print("-" * 75)
    
    for n in n_range:
        # 创建Hilbert矩阵
        H = create_hilbert_matrix(n)
        
        # 使用NumPy计算条件数
        start_time = time.time()
        np_cond = np.linalg.cond(H)
        np_time = time.time() - start_time
        cond_numpy.append(np_cond)
        times_numpy.append(np_time)
        
        # 使用估计方法计算条件数
        start_time = time.time()
        est_cond = condition_number(H)
        est_time = time.time() - start_time
        cond_estimate.append(est_cond)
        times_estimate.append(est_time)
        
        # 打印结果
        print(f"{n:<5}{np_cond:<20.4e}{est_cond:<20.4e}{np_time:<15.6f}{est_time:<15.6f}")
    
    # 绘制条件数比较图
    plt.figure(figsize=(12, 10))
    
    # 条件数比较
    plt.subplot(2, 1, 1)
    plt.semilogy(n_range, cond_numpy, 'o-', label='NumPy Condition Number')
    plt.semilogy(n_range, cond_estimate, 's--', label='Estimated Condition Number')
    plt.grid(True)
    plt.xlabel('Hilbert Matrix Order')
    plt.ylabel('Condition Number (log scale)')
    plt.title('Condition Number Comparison for Hilbert Matrices')
    plt.legend()
    
    # 计算时间比较
    plt.subplot(2, 1, 2)
    plt.plot(n_range, times_numpy, 'o-', label='NumPy Time')
    plt.plot(n_range, times_estimate, 's--', label='Estimation Algorithm Time')
    plt.grid(True)
    plt.xlabel('Hilbert Matrix Order')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hilbert_condition_numbers.png')
    plt.show()
    
    return n_range, cond_numpy, cond_estimate


if __name__ == "__main__":
    print("Computing condition numbers for Hilbert matrices (order 5-20)...\n")
    compare_condition_numbers()
    print("\nComputation complete. Results saved as 'hilbert_condition_numbers.png'")