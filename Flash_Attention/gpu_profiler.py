import torch
import time
import os
import shutil

from torch.utils.tensorboard import SummaryWriter


class GPUProfiler:
    def __init__(self, logfile = "gpu_profile.log", reset=True, tensorboard_logdir="./tb_logs"):
        self.enabled = torch.cuda.is_available()
        self.logfile = logfile
        if reset:
            open(self.logfile, "w").close() 
            
        if reset and tensorboard_logdir is not None:
            if os.path.exists(tensorboard_logdir):
                shutil.rmtree(tensorboard_logdir)
            
        self.writer = None
        if tensorboard_logdir is not None:
            self.writer = SummaryWriter(log_dir=tensorboard_logdir)
        
        self.step = 0 

    
    def _log(self, msg):
        with open(self.logfile, "a") as f:
            f.write(msg + "\n")
            
    def start(self):
        if self.enabled:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_time = time.time()
            self.start_mem = torch.cuda.memory_allocated()

    def end(self, name):
        if self.enabled:
            torch.cuda.synchronize()
            end_time = time.time()
            end_mem = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()

            msg = (
                f"[{name:<20}] "
                f"Time: {(end_time - self.start_time)*1000:.2f} ms | "
                f"VRAM_d: {(end_mem - self.start_mem)/1024**2:.2f} MB | "
                f"Total: {end_mem/1024**2:.2f} MB | "
                f"Peak: {peak/1024**2:.2f} MB"
            )
            
            self._log(msg)
            
            if self.writer:
                self.writer.add_scalar(f'{name}/VRAM_delta_MB', (end_mem - self.start_mem)/1024**2, self.step)
                self.writer.add_scalar(f'{name}/VRAM_total_MB', end_mem/1024**2, self.step)
                self.writer.add_scalar(f'{name}/VRAM_peak_MB', peak/1024**2, self.step)
                self.step += 1
