# Seed-VC Real-Time Benchmark Tools

Comprehensive benchmarking suite for validating real-time voice conversion feasibility on different hardware configurations.

## 📋 Overview

These tools help you:
1. **Auto-detect** your system specifications (Apple Silicon, GPU, memory)
2. **Benchmark** all model configurations with detailed timing
3. **Validate** real-time feasibility with statistical analysis
4. **Identify bottlenecks** with per-component latency breakdown
5. **Compare** models and find optimal configurations for your hardware

## 🛠️ Tools

### 1. Full Benchmark (`benchmark_realtime.py`)

Complete benchmarking suite with detailed analysis.

```bash
# Benchmark all models
python tools/benchmark_realtime.py

# Benchmark specific model
python tools/benchmark_realtime.py --model xlsr_tiny

# Custom configuration
python tools/benchmark_realtime.py \
    --model xlsr_tiny \
    --steps 4,6,8,10,20 \
    --block-time 0.20 \
    --runs 10

# Save results
python tools/benchmark_realtime.py --output my_results.json
```

**Options:**
- `--model, -m`: Model to benchmark (`xlsr_tiny`, `whisper_small`, `whisper_base`)
- `--steps, -s`: Diffusion steps to test (comma-separated, default: `4,6,8,10`)
- `--block-time, -b`: Block time in seconds (default: `0.20`)
- `--runs, -r`: Number of benchmark runs per config (default: `5`)
- `--test-audio`: Path to test audio file
- `--reference-audio`: Path to reference audio file
- `--output, -o`: Output JSON file path
- `--verbose, -v`: Verbose output

### 2. Quick Benchmark (`quick_benchmark.py`)

Fast feasibility check for rapid testing.

```bash
# Quick check (3 runs, single config)
python tools/quick_benchmark.py

# Test specific model
python tools/quick_benchmark.py --model whisper_small --steps 4,6

# Custom block time
python tools/quick_benchmark.py --block-time 0.25
```

**Options:**
- Same as full benchmark but optimized for speed (3 runs default)

## 📊 Output Format

### Console Output

```
================================================================================
SYSTEM SPECIFICATIONS
================================================================================
OS: Darwin 23.3.0
Chip: Apple M4 Max
CPU: 12 physical / 16 logical
Memory: 48.0GB total, 42.1GB available
PyTorch: 2.1.0
Device: mps
MPS: Available ✅
================================================================================

🚀 Benchmarking xlsr_tiny...
   Description: 25M DiT + XLSR-large (300M frozen) + HiFi-GAN

✅ xlsr_tiny (6 steps)
------------------------------------------------------------
Block time: 0.20s (200ms)
Latency - Mean: 137ms, P50: 135ms, P90: 142ms, P99: 148ms
RTF: 0.685 (Real-Time Factor)
Component breakdown:
  - Content Encoder: 45ms
  - DiT Inference: 58ms
  - Vocoder: 22ms
  - Length Regulator: 8ms
  - VAD: 4ms
Feasibility - Mean < 80% block: ✅, P99 < 100% block: ✅

================================================================================
BENCHMARK SUMMARY
================================================================================
Model                  Steps    Block   Mean(ms)    P99(ms)      RTF   Feasible
--------------------------------------------------------------------------------
xlsr_tiny                  6     0.20s       137       148    0.685         ✅
xlsr_tiny                  8     0.20s       178       185    0.890         ❌
xlsr_tiny                 10     0.20s       215       228    1.075         ❌
================================================================================

Feasible configurations: 1/3
Best configuration: xlsr_tiny with 6 steps (RTF: 0.685, Latency: 137ms)

✅ VERDICT: xlsr_tiny is FEASIBLE for real-time
   Recommended: 6 steps (Latency: 137ms, RTF: 0.685)
```

### JSON Output

```json
{
  "timestamp": "2024-03-30T14:23:45.123456",
  "model_name": "xlsr_tiny",
  "diffusion_steps": 6,
  "block_time": 0.20,
  "num_runs": 5,
  "latencies_ms": [135.2, 137.8, 136.5, 138.1, 137.4],
  "mean_latency_ms": 137.0,
  "std_latency_ms": 1.2,
  "min_latency_ms": 135.2,
  "max_latency_ms": 138.1,
  "p50_latency_ms": 137.0,
  "p90_latency_ms": 138.0,
  "p99_latency_ms": 148.0,
  "rtf": 0.685,
  "component_latencies": {
    "vad": 4.2,
    "content_encoder": 45.3,
    "length_regulator": 8.1,
    "dit": 58.4,
    "vocoder": 22.0,
    "total": 137.0
  },
  "feasible": true,
  "feasible_mean": true,
  "feasible_p99": true,
  "system_specs": {
    "chip": "Apple M4 Max",
    "device": "mps",
    "memory_total_gb": 48.0
  }
}
```

## 🎯 Feasibility Criteria

A configuration is marked as **feasible** for real-time use if:

1. **Mean latency < 80% of block time** (20% headroom for variability)
2. **P99 latency < 100% of block time** (even worst cases fit)
3. **RTF (Real-Time Factor) < 0.8**

### Why These Criteria?

- **80% mean threshold**: Accounts for system variability, GC pauses, background processes
- **P99 threshold**: Ensures 99% of blocks process in time (only 1% may lag)
- **RTF < 0.8**: Real-Time Factor = latency / block_time, should be comfortably below 1.0

## 📈 Understanding Results

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Mean Latency** | Average processing time | < 80% block time |
| **P99 Latency** | 99th percentile (worst case) | < 100% block time |
| **RTF** | Real-Time Factor (latency/block) | < 0.8 |
| **Std Dev** | Latency variability | Lower is better |

### Component Breakdown

- **VAD**: Voice Activity Detection latency
- **Content Encoder**: XLSR/Whisper feature extraction
- **Length Regulator**: Sequence length adaptation
- **DiT**: Main diffusion model inference (scales with steps)
- **Vocoder**: Audio synthesis (HiFi-GAN/BigVGAN)

### Interpreting Bottlenecks

```
If content_encoder > 50ms:
    → Consider using smaller encoder (XLSR vs Whisper)
    
If dit > 100ms:
    → Reduce diffusion steps
    → Consider smaller model (xlsr_tiny vs whisper_small)
    
If vocoder > 30ms:
    → HiFi-GAN is faster than BigVGAN
    → Consider reducing output quality
```

## 🔧 System Detection

The benchmark auto-detects:

- **Apple Silicon chip** (M1/M2/M3/M4 series)
- **GPU cores** (physical and logical)
- **Unified memory** (total and available)
- **PyTorch device** (MPS/CUDA/CPU)
- **MPS availability** for Apple Silicon

### Example Detections

```
Chip: Apple M4 Max           → High-end, expect good performance
Chip: Apple M2               → Mid-range, may need optimizations
Chip: Apple M1               → Entry-level, use smallest configs
```

## 🧪 Test Audio Requirements

### Test Audio (Source)
- **Duration**: 30-60 seconds recommended
- **Format**: WAV, MP3, FLAC, etc.
- **Content**: Clear speech, minimal background noise
- **Default**: `examples/source/source_1.wav`

### Reference Audio (Target Voice)
- **Duration**: At least 3 seconds
- **Format**: Same as test audio
- **Content**: Clean voice sample of target speaker
- **Default**: `examples/reference/ref_1.wav`

## 📊 Benchmark Scenarios

### Scenario 1: Quick Feasibility Check

```bash
# Is xlsr_tiny feasible on my system?
python tools/quick_benchmark.py --model xlsr_tiny --steps 6
```

**Expected output**: 30-60 seconds

### Scenario 2: Find Optimal Steps

```bash
# What's the best diffusion steps for my hardware?
python tools/benchmark_realtime.py \
    --model xlsr_tiny \
    --steps 4,6,8,10,12,15,20
```

**Expected output**: 3-5 minutes

### Scenario 3: Compare All Models

```bash
# Which models are feasible on my system?
python tools/benchmark_realtime.py --steps 4,6,8
```

**Expected output**: 10-20 minutes

### Scenario 4: Production Validation

```bash
# Validate production config with many runs
python tools/benchmark_realtime.py \
    --model xlsr_tiny \
    --steps 6 \
    --block-time 0.20 \
    --runs 20 \
    --output production_validation.json
```

**Expected output**: 5-10 minutes, comprehensive statistics

## 🐛 Troubleshooting

### "No GPU available"

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Install PyTorch with MPS/CUDA support

### "Checkpoint not found"

```bash
# Download models manually
python -c "from hf_utils import load_custom_model_from_hf; load_custom_model_from_hf('Plachta/Seed-VC', 'DiT_uvit_tat_xlsr_ema.pth', None)"
```

### "Memory error"

```bash
# Close other applications
# Reduce number of runs
python tools/quick_benchmark.py --runs 2

# Use smaller model
python tools/quick_benchmark.py --model xlsr_tiny
```

### "Model not feasible"

```bash
# Try fewer steps
python tools/quick_benchmark.py --model xlsr_tiny --steps 4

# Increase block time (higher latency but more feasible)
python tools/quick_benchmark.py --block-time 0.30
```

## 📁 Output Files

### JSON Results

Saved as `benchmark_results_{chip}_{timestamp}.json` containing:
- All benchmark results
- System specifications
- Component latencies
- Feasibility verdicts

### Usage

```python
import json

# Load results
with open('benchmark_results_Apple_M4_Max_20240330_142345.json') as f:
    results = json.load(f)

# Find feasible configs
feasible = [r for r in results if r['feasible']]

# Get best config
best = min(feasible, key=lambda x: x['rtf'])
print(f"Best: {best['model_name']} with {best['diffusion_steps']} steps")
```

## 🎓 Best Practices

1. **Start with quick benchmark** to get fast feedback
2. **Use full benchmark** for production validation
3. **Test multiple step counts** to find optimal quality/speed tradeoff
4. **Run enough iterations** (≥5 for statistics, ≥20 for production)
5. **Save results** for comparison across hardware/configurations
6. **Monitor component latencies** to identify optimization targets

## 📚 References

- [Seed-VC README](https://github.com/Plachtaa/Seed-VC)
- [FINETUNING_AND_REALTIME_OPTIMIZATION.md](../docs/FINETUNING_AND_REALTIME_OPTIMIZATION.md)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## 🤝 Contributing

Found issues or have suggestions? Please:
1. Run the benchmark on your system
2. Save the JSON results
3. Open an issue with your results and system specs

## 📄 License

Same as Seed-VC project
