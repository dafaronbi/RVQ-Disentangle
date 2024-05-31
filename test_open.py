import torch
from datetime import datetime
import threading

def multi_threaded_file_reader(file_paths):
    threads = []
    results = []

    # Define the worker function
    def read_file_thread(file_path):
        result = torch.load(file_path)
        results.append(result)

    # Create and start threads
    for file_path in file_paths:
        thread = threading.Thread(target=read_file_thread, args=(file_path,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return results

fps = ["train_tensor_" + str(i) + ".pt" for i in range(12)]

print(f"LOADING TENSOR...")
start_time = datetime.now()
x = multi_threaded_file_reader(fps)
end_time = datetime.now()
print(f"DONE")
print(f"Load Time: {end_time-start_time}")