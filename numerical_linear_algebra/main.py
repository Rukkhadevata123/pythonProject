import lu_decomposition
import numpy as np

A = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 10]])
b = np.array([1, 1, 1])

L, U = lu_decomposition.lu_decomposition(A)
x = lu_decomposition.solve_linear_system(A, b)
print(L)
print(U)
print(x)

print('----------------------')

L, U, P = lu_decomposition.lu_decomposition_with_partial_pivoting(A)
x = lu_decomposition.solve_with_partial_pivoting(A, b)
print(L)
print(U)
print(P)
print(x)

print('----------------------')

L, U, P, Q = lu_decomposition.lu_decomposition_with_complete_pivoting(A)
x = lu_decomposition.solve_with_complete_pivoting(A, b)
print(L)
print(U)
print(P)
print(Q)
print(x)