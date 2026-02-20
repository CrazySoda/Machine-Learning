import torch
import time
import os
import shutil
import json
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter


class GPUProfiler:
    def __init__(self, logfile="gpu_profile.log", reset=True, tensorboard_logdir="./tb_logs"):
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

        # ---- Structured metric collection ----
        # Accumulates per-layer stats within an epoch
        self._layer_times = defaultdict(list)     # name -> [time_ms, ...]
        self._layer_mem_deltas = defaultdict(list)  # name -> [mem_delta_MB, ...]
        self._layer_peaks = defaultdict(list)      # name -> [peak_MB, ...]
        self.collecting = False  # toggle on/off for structured collection

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

            time_ms = (end_time - self.start_time) * 1000
            mem_delta_mb = (end_mem - self.start_mem) / 1024 ** 2
            total_mb = end_mem / 1024 ** 2
            peak_mb = peak / 1024 ** 2

            msg = (
                f"[{name:<20}] "
                f"Time: {time_ms:.2f} ms | "
                f"VRAM_d: {mem_delta_mb:.2f} MB | "
                f"Total: {total_mb:.2f} MB | "
                f"Peak: {peak_mb:.2f} MB"
            )

            self._log(msg)

            if self.writer:
                self.writer.add_scalar(f'{name}/VRAM_delta_MB', mem_delta_mb, self.step)
                self.writer.add_scalar(f'{name}/VRAM_total_MB', total_mb, self.step)
                self.writer.add_scalar(f'{name}/VRAM_peak_MB', peak_mb, self.step)
                self.step += 1

            # Collect structured data when enabled
            if self.collecting:
                self._layer_times[name].append(time_ms)
                self._layer_mem_deltas[name].append(mem_delta_mb)
                self._layer_peaks[name].append(peak_mb)

    # ---- Epoch-level metric helpers ----

    def start_epoch_collection(self):
        """Call at the start of each epoch to begin collecting per-layer stats."""
        self.collecting = True
        self._layer_times.clear()
        self._layer_mem_deltas.clear()
        self._layer_peaks.clear()

    def end_epoch_collection(self):
        """
        Call at the end of each epoch.
        Returns a dict summarising mean time_ms and mean mem_delta_MB per layer name.
        """
        self.collecting = False
        summary = {}
        all_names = set(self._layer_times.keys()) | set(self._layer_mem_deltas.keys())
        for name in sorted(all_names):
            times = self._layer_times.get(name, [])
            mems = self._layer_mem_deltas.get(name, [])
            peaks = self._layer_peaks.get(name, [])
            summary[name] = {
                "mean_time_ms": sum(times) / len(times) if times else 0.0,
                "total_time_ms": sum(times),
                "mean_mem_delta_MB": sum(mems) / len(mems) if mems else 0.0,
                "mean_peak_MB": sum(peaks) / len(peaks) if peaks else 0.0,
                "calls": len(times),
            }
        return summary

    @staticmethod
    def save_metrics(metrics: dict, filepath: str):
        """Save a metrics dictionary to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
