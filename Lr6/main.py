import timeit
from fermat_cython import fermat_factorization

TEST_LST = [101, 9973, 104729, 101909, 609133, 1300039, 9999991, 99999959, 99999971, 3000009, 700000133, 61335395416403926747]

if __name__ == '__main__':
  time_res = timeit.repeat("res = [fermat_factorization(i) for i in TEST_LST]",
  setup = 'from __main__ import fermat_factorization, TEST_LST',
  number = 10,
  repeat = 1
)
print(time_res)
print(f"Среднее время: {sum(time_res)/len(time_res):.4f} сек")
  
# = [101, 9973, 104729, 101909, 609133, 1300039, 9999991, 99999959, 99999971, 3000009, 700000133, 61335395416403926747];'