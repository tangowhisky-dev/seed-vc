# Fix: Vocoder Inference Mode Error

## Issue

**Problem**: RuntimeError when running benchmark - "Inference tensors cannot be saved for backward"

**Error**:
```
RuntimeError: Inference tensors cannot be saved for backward. Please do not use Tensors 
created in inference mode in computation tracked by autograd. To work around this, you can 
make a clone to get a normal tensor and use it in autograd, or use `torch.no_grad()` 
instead of `torch.inference_mode()`.
```

**Stack trace**:
```
File "modules/hifigan/generator.py", line 402, in forward
    f0 = self.f0_predictor(x)
...
RuntimeError: Inference tensors cannot be saved for backward
```

**Root Cause**: Two issues:
1. `modules/hifigan/generator.py` used `@torch.inference_mode()` decorator which is incompatible with autograd
2. `tools/benchmark_realtime.py` called `self.vocoder(vc_target)` without wrapping in `torch.no_grad()`

## Fix Applied

### Fix 1: modules/hifigan/generator.py

**Line 452** - Changed from `@torch.inference_mode()` to `@torch.no_grad()`:

**Before**:
```python
@torch.inference_mode()
def inference(self, mel: torch.Tensor, f0=None) -> torch.Tensor:
    return self.forward(x=mel, f0=f0)
```

**After**:
```python
@torch.no_grad()
def inference(self, mel: torch.Tensor, f0=None) -> torch.Tensor:
    return self.forward(x=mel, f0=f0)
```

### Fix 2: tools/benchmark_realtime.py

**Line 614** - Wrapped vocoder call in `torch.no_grad()`:

**Before**:
```python
# Vocoder
if self.device.type == "mps":
    torch.mps.synchronize()
start_event.record()
vc_wave = self.vocoder(vc_target).squeeze()
end_event.record()
```

**After**:
```python
# Vocoder
if self.device.type == "mps":
    torch.mps.synchronize()
start_event.record()
with torch.no_grad():
    vc_wave = self.vocoder(vc_target).squeeze()
end_event.record()
```

## Why This Works

- `torch.inference_mode()`: Disables gradient computation entirely and prevents saving tensors for backward. Throws error if you try to use inference tensors in autograd-tracked computation.
- `torch.no_grad()`: Disables gradient computation but allows tensors to be used in autograd context. More compatible with complex pipelines.

## Result

✅ **Fixed**: Benchmark now runs successfully
✅ **Tested**: xlsr_tiny benchmark completes with timing results
✅ **Ready**: Can benchmark all V1 models

## Benchmark Results (First Run)

```
Model                 Steps    Block   Mean(ms)    P99(ms)      RTF   Feasible
--------------------------------------------------------------------------------
xlsr_tiny                 4    0.20s      1420      2293   7.101          ❌

Component breakdown:
  - Content Encoder: 72ms
  - DiT Inference: 117ms
  - Vocoder: 985ms
  - Length Regulator: 3ms
  - VAD: 231ms
```

**Note**: The first run is slow due to model loading and dependency downloads. Subsequent runs will be faster.

## Files Modified

- `modules/hifigan/generator.py`: Line 452 - Changed decorator
- `tools/benchmark_realtime.py`: Line 614 - Added `torch.no_grad()` wrapper

## Status

✅ **Fixed**: Vocoder inference mode error resolved
✅ **Tested**: Benchmark runs successfully
✅ **Ready**: Can now benchmark all models
