import time 
import os 
import psutil
import matplotlib.pyplot as plt

from .utils import synchronize

class TimeProfiler:
    def __init__(self):
        self.time = None
        self.start = None 
        self.scopes = {}

    def __enter__(self):
        synchronize()
        self.start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        synchronize()
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        self.time = time.perf_counter() - self.start
        return True

    def scope(self, name):
        assert name not in self.scopes, f"Scope {name} already exists"
        self.scopes[name] = None
        return TimeProfilerScope(self, name)
    
    def plot(self, save_path="time.png"):
        assert len(self.scopes) > 0, "No scopes to plot"
        fig, ax = plt.subplots(figsize=(8,6))
        ax.broken_barh([(0, self.time)], (0, len(self.scopes)), facecolors='blue', alpha=0.2)
        for i,(scope, (start, end)) in enumerate(self.scopes.items()):
            ax.broken_barh([(start-self.start, end-start)], (i,1), facecolors='orange')
            ax.text((start-self.start+end-self.start)/2, i+0.5, scope, ha="center", va="center", color="black", fontsize=12)
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("Scope")
        ax.set_yticks([])
        fig.savefig(save_path)


class TimeProfilerScope:
    def  __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        synchronize()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is KeyboardInterrupt:
            return True 
        elif exc_type is not None:
            return False 
        
        synchronize()
        self.end_time = time.perf_counter()
    
        self.profiler.scopes[self.name] = (self.start_time, self.end_time)
        return True