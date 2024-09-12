import torch
import multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm
import time

multiprocessing.set_start_method('spawn', force=True)

def worker(gpu_id, queue):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        # 创建一个张量并将其移动到指定的GPU上
        tensor = torch.rand(100, 100).to(device)
        
        # 模拟耗时操作
        for i in tqdm(range(50), desc=f"Worker {gpu_id}", leave=True, position=gpu_id):
            time.sleep(0.1)  # 模拟耗时操作
            queue.put((gpu_id, i + 1))

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No CUDA-enabled device found.")
    else:
        num_gpus = torch.cuda.device_count()
        processes = []
        progress_queue = Queue()

        # 启动进程
        for i in range(num_gpus):
            p = Process(target=worker, args=(i, progress_queue))
            p.start()
            processes.append(p)

        # 创建一个总进度条来显示所有进程的进度
        total_steps = sum([50] * num_gpus)
        # with tqdm(total=total_steps, desc="Total Progress", leave=True) as pbar:
        #     while True:
        #         try:
        #             # 从队列中获取进度信息
        #             gpu_id, step = progress_queue.get(timeout=1)
        #             pbar.update(1)
        #             pbar.set_postfix_str(f"GPU {gpu_id}: {step}/50")
        #         except Exception as e:
        #             # 当没有新的更新时，检查所有进程是否已经结束
        #             if all(p.exitcode is not None for p in processes):
        #                 break
        while True:
            if all(p.exitcode is not None for p in processes):
                break

        # 确保所有进程完成
        for p in processes:
            p.join()