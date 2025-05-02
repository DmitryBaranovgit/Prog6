import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from fermat_cython import fermat_factorization

TEST_LST = [101, 9973, 104729, 101909, 609133, 1300039, 9999991, 99999959, 99999971, 3000009, 700000133, 61335395416403926747]

def parallel_submit_mode(executor_class, data, func, max_workers = 4, label = ""):
    """Режим распределения задач через sybmit + as_completed"""
    start = time.time()
    with executor_class(max_workers = max_workers) as executor:
        spawn = partial(executor.submit, func)
        futures = [spawn(i) for i in data]
        results = [f.result() for f in as_completed(futures)]
    end = time.time()
    print(f"[{label}] Время выполнения (submit): {end - start:.4f} секунд")
    return results, end - start

def parallel_map_mode(executor_class, data, func, max_workers = 4, label = ""):
    """Режим map (удобнее, если не нужен доступ к отдельным future-объектам)"""
    start = time.time()
    with executor_class(max_workers = max_workers) as executor:
        results = list(executor.map(func, data))
    end = time.time()
    print(f"[{label}] Время выполнения (map): {end - start:.4f} секунд")
    return results, end - start

if __name__ == '__main__':
    print("=== Распределённое факторизирование методом Ферма ===")

    print("\n--- Потоки ---")
    results_thread_submit, t1 = parallel_submit_mode(ThreadPoolExecutor, TEST_LST, fermat_factorization, label = "Потоки")
    results_thread_map, t2 = parallel_map_mode(ThreadPoolExecutor, TEST_LST, fermat_factorization, label = "Потоки")

    print("\n--- Процессы ---")
    results_proc_submit, p1 = parallel_submit_mode(ProcessPoolExecutor, TEST_LST, fermat_factorization, label = "Процессы")
    results_proc_map, p2 = parallel_map_mode(ProcessPoolExecutor, TEST_LST, fermat_factorization, label = "Процессы")

    print("\n ---Сравнение---")
    print(f"Потоки (submit): {t1:.4f}s, Потоки (map): {t2:.4f}s")
    print(f"Процессы (submit): {p1:.4f}s, Процессы (map): {p2:.4f}s")