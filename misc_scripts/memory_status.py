import psutil
import GPUtil

# CPU info
print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
print(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
print(f"CPU usage per core: {psutil.cpu_percent(percpu=True)}")
print(f"Total memory (GB): {psutil.virtual_memory().total / 1e9:.2f}")

# GPU info
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load*100:.1f}%, Memory Free: {gpu.memoryFree}MB, Memory Total: {gpu.memoryTotal}MB")
