# Print Toeplitz matrices
from scipy.linalg import toeplitz, hankel

print("Toeplitz")
print(toeplitz(c=[1, 2 + 1.0j, 3]))

print("")

print("Hankel")
print(hankel(c=[1, 2 + 1.0j, 3]))
