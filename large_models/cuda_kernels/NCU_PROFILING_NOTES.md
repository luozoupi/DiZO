# NCU Profiling Notes

## Issue: NCU Profile Files Not Found

### Root Cause

The ncu profiler was using `--set full` which profiles every kernel with extensive metrics, causing:
1. Very long profiling time (60+ seconds)
2. Potential timeouts
3. File may not be created if profiling fails

### Solution

The script has been updated to use `--set default` instead of `--set full` for faster, more practical profiling.

### File Location

Note: The script uses `SCRIPT_DIR` which resolves to `/mnt/data1/luo00466/...` even though you may be working in `/home/luo00466/luo00466_data1/...`. This is because `/home/luo00466/luo00466_data1` is a symlink to `/mnt/data1/luo00466`.

**Both paths point to the same files**, so you can access them via either:
- `/home/luo00466/luo00466_data1/DiZO_old/DiZO/large_models/cuda_kernels/benchmark_results/profiler_logs/ncu/opt-350m/`
- `/mnt/data1/luo00466/DiZO_old/DiZO/large_models/cuda_kernels/benchmark_results/profiler_logs/ncu/opt-350m/`

### NCU Options

If you want more detailed profiling, you can modify the ncu command in the script:

```python
# For faster profiling (default):
'--set', 'default'

# For detailed profiling (slower):
'--set', 'full'

# For very fast profiling (fewer metrics):
'--set', 'speed-of-light'

# To profile only specific kernels:
'--kernel-regex', 'kernel_name_pattern'
```

### Permissions

NCU typically does NOT require sudo. If you get permission errors, it's usually because:
1. The GPU is in use by another process
2. The CUDA driver version is incompatible
3. There's a system configuration issue

### Verifying NCU Works

```bash
# Test ncu on a simple script
echo "import torch; x=torch.randn(1000,1000).cuda(); y=x@x" > test_ncu.py
ncu --set default python test_ncu.py
```

If this works, ncu is functioning correctly and the issue was likely the `--set full` timeout.

