# Machine-Learning

## Machine Learning GPU Profiler

### Running the GPU Log Monitor on Windows

Before running the GPU log monitoring script (`watch_gpu.ps1`), you need to allow PowerShell scripts to execute on your system.

1. **Open PowerShell as Administrator**
2. Run the following command to allow local scripts to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

3. Then run the GPU log monitor script:

```powershell
.\watch_gpu.ps1
```

### Viewing Logs on TensorBoard

You can observe the plots on TensorBoard after starting `train_test.py`. Run:

```powershell
tensorboard --logdir=./tb_logs
```

