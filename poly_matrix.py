import numpy as np
class polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def eval(self, t):
      result = 0
      for coeff in reversed(self.coeffs):
          result = result * t + coeff
      return np.max(result, 0)

    def __repr__(self):
        return " + ".join(f"{coeff} * x^{i}" for i, coeff in enumerate(self.coeffs))

def create_poly_matrix(N, k, time_horizon):
    """
    Create a polynomial matrix of size N x N with random coefficients.
    Each entry is a polynomial of degree k.
    """
    poly_matrix = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            if i == j:
                poly_matrix[i, j] = polynomial([0])
                continue
            else:
              c0 = (time_horizon - time_horizon ** (k + 1)) / (1 - time_horizon)
              coeffs = np.random.uniform(low = -1, high = 1, size = k)
              coeffs = np.insert(coeffs, 0, c0)
              poly_matrix[i, j] = polynomial(coeffs)
    return poly_matrix