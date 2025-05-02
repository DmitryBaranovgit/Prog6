from math import isqrt

cpdef bint is_perfect_square(object n):
  cdef object root = isqrt(n)
  return root * root == n

cpdef tuple fermat_factorization(object n):
  if n % 2 == 0:
    return 2, n // 2

  x = isqrt(n) + 1
  while True:
    y_squared = x * x - n
    if is_perfect_square(y_squared):
      y = isqrt(y_squared)
      return x - y, x + y
    x += 1