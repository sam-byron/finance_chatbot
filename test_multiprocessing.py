from multiprocessing import Pool
import os
import time

def load_chunk(path):
    print(f"PID {os.getpid()} processing {path}")
    time.sleep(2)
    return f"done {path}"

if __name__ == "__main__":
    chunk_paths = [f"chunk_{i}" for i in range(8)]
    max_workers = min(len(chunk_paths), 4)

    with Pool(processes=max_workers) as pool:
        results = pool.map(load_chunk, chunk_paths)

    print("Results:", results)
