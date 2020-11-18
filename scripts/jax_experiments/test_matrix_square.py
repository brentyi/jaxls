import numpy as np

# residuals, variables
A = np.zeros((3, 5))


A[0:3, :2] = 1.0
A[0:3, 4:] = 0.5

x = np.ones(5)

ATA = A.T @ A

print(A.T)
print()
print(A)
print()
print(ATA)
print()
print(ATA @ x)
