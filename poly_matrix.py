import numpy as np
class polynomial:
  def __init__(self, roots, constant, scale, zero = False):
      self.roots = roots
      self.zero = zero
      self.constant = constant
      self.scale = scale

  def eval(self, t):
    if self.zero:
        return 20
    else:
      result = -1
      for root in self.roots:
          result *= (t - root)
      result += self.constant
      result *= self.scale
      return max(result, 0)

  def __repr__(self):
      return " + ".join(f"{coeff} * x^{i}" for i, coeff in enumerate(self.roots))



#   Create a polynomial matrix of size N x N with random coefficients.
#   Each entry is a polynomial of degree k.

def create_poly_matrix(N, time_horizon, seed=50):

    if seed is not None:
        np.random.seed(seed)
    
  
    poly_matrix = np.zeros((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            if i == j:
                poly_matrix[i, j] = polynomial([], 0, 0, zero = True)
                continue
            else:
                r1 = np.random.normal(-1, 3)
                r2 = np.random.normal(5, 3)
                r3 = np.random.normal(20, 3)
                r4 = np.random.normal(25, 3)
                r5 = np.random.normal(9, 3)
                r6 = np.random.normal(13, 3)
                roots = (r1, r2, r3, r4, r5, r6)
                constant = 1.2 * -np.min([-1 * (x - r1) * (x - r2) * (x - r3) * (x - r4) * (x - r5) * (x - r6) for x in range(time_horizon)])
                scale = np.random.uniform(0.0, 1.0) / 500_000
                poly_matrix[i, j] = polynomial(roots, constant, scale)
    return poly_matrix