import time
import tracemalloc

class MetricsTracker:
    """
    Rastreia o tempo de execução (Wall Clock) e consumo de memória RAM (Peak Memory).
    """

    @staticmethod
    def run_with_profiling(func, *args, **kwargs):
        """
        Executa uma função medindo o tempo total e o pico de alocação de memória RAM.

        Returns:
            tuple: (result, wall_time_sec, peak_mem_mb)
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        wall_time_sec = float(end_time - start_time)
        peak_mem_mb = float(peak_bytes / (1024.0 * 1024.0))

        return result, wall_time_sec, peak_mem_mb