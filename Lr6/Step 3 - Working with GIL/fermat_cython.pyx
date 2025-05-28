from libc.math cimport sqrt
from cython.cimports.libc.stdlib import malloc, free
from cython cimport nogil, gil

cpdef bint is_perfect_square(unsigned long long n):
    cdef unsigned long long root = <unsigned long long> sqrt(n)
    return root * root == n or (root + 1) * (root + 1) == n

cpdef tuple fermat_factorization(unsigned long long n):
    if n % 2 == 0:
        return 2, n // 2

    cdef unsigned long long x = <unsigned long long> sqrt(n) + 1
    cdef unsigned long long y_squared
    cdef unsigned long long y = 0

    with nogil:
        while True:
            y_squared = x * x - n
            with gil:
                if is_perfect_square(y_squared):
                    y = <unsigned long long> sqrt(y_squared)
                    break
            x += 1
    
    return x - y, x + y