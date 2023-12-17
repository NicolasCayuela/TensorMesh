import os 
import psutil
import multiprocessing
import time 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

def get_memory_for_pid(pid):
    process = psutil.Process(pid)
    memory_usage = process.memory_info().rss / (1024 ** 2)
    for child in process.children(recursive=True):
        memory_usage += child.memory_info().rss / (1024 ** 2)
    return memory_usage

def monitor_cpu_memory(pid, stop_event, conn):
    result = []
    times  = []
    start_time = time.perf_counter()
    while not stop_event.is_set():
        memory = get_memory_for_pid(pid)
        result.append(memory)
        times.append(time.perf_counter() - start_time)

    conn.send(tuple(result))
    conn.send(tuple(times))
    conn.close()


class CPUProfiler:
    def __init__(self):
        self.pid = os.getpid()
        self.scopes = {}
    
    def __enter__(self):
        self.stop_event = multiprocessing.Event()
        parent_conn, child_conn = multiprocessing.Pipe()
        self.conn = parent_conn
        self.monitoring_process = multiprocessing.Process(
            target=monitor_cpu_memory, 
            args=(self.pid, self.stop_event, child_conn))
        self.start_mem  = get_memory_for_pid(self.pid)
        self.start_time = time.perf_counter() 
        self.monitoring_process.start()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is KeyboardInterrupt:
            self.stop_event.set()
            self.monitoring_process.join()
            return True 
        elif exc_type is not None:
            return False 
        
        self.stop_event.set()
        results = self.conn.recv()
        times   = self.conn.recv()
        self.monitoring_process.join()

        self.results = tuple(result-self.start_mem for result in results) 
        self.times   = times
        return True
    
    def max(self):
        return max(self.results)

    def min(self):
        return min(self.results)

    def mean(self):
        return np.mean(self.results)
    
    def std(self):
        return np.std(self.results)
       
    def __array__(self):
        return np.array(self.results)

    def plot(self, save_path="cpu_mem.png"):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(self.times,self.results)
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel(f"Used CPU Memory (MB)")

        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        max_mem = self.max()
        min_mem = self.min()
        one_third = (max_mem - min_mem) / 3 + min_mem
        two_third = (max_mem - min_mem) / 3 * 2 + min_mem
        for i,(name, (start, end)) in enumerate(self.scopes.items()):
            is_odd = i % 2 == 1
            ax.axvspan(start-self.start_time, end-self.start_time, alpha=0.3, color=colors[i], label=name)
            ax.text((start-self.start_time+end-self.start_time)/2, one_third if is_odd else two_third, name, ha="center", va="center", color="black", fontsize=12)

        fig.savefig(save_path)
        plt.close()

    def scope(self, name):
        assert name not in self.scopes, f"Scope {name} already exists"
        self.scopes[name] = None
        return CPUProfilerScope(self, name)


class CPUProfilerScope:
    def  __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        
        self.end_time = time.perf_counter()
    
        self.profiler.scopes[self.name] = (self.start_time, self.end_time)
        return True


