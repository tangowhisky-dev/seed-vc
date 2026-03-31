# Seed-VC Real-Time Benchmark Implementation

## Executive Summary

This document describes the implementation of a comprehensive benchmarking suite for validating real-time voice conversion feasibility on different hardware configurations, particularly Apple Silicon Macs.

## Problem Statement

The existing analysis in `docs/FINETUNING_AND_REALTIME_OPTIMIZATION.md` claims that **only the smallest model (xlsr-tiny, ~25M DiT parameters) is feasible for real-time operation** on Mac hardware. This claim was based on:

1. **Theoretical FLOP calculations**
2. **Estimated latency breakdowns**
3. **MPS performance assumptions**

However, **no empirical validation** existed to confirm these claims on actual hardware.

## Solution: Benchmark Suite

We implemented a three-tier benchmarking solution:

### 1. Full Benchmark (`benchmark_realtime.py`)

**Purpose**: Comprehensive validation with detailed statistics

**Features**:
- ✅ Auto-detects system specs (Apple Silicon chip, GPU cores, memory)
- ✅ Benchmarks all model configurations (xlsr_tiny, whisper_small, whisper_base)
- ✅ Measures per-component latency (VAD, encoder, DiT, vocoder, etc.)
- ✅ Computes statistical metrics (mean, std, P50, P90, P99)
- ✅ Calculates RTF (Real-Time Factor)
- ✅ Validates feasibility with strict criteria
- ✅ Exports results to JSON for analysis

**Usage**:
```bash
# Benchmark all models
python tools/benchmark_realtime.py

# Specific model with custom steps
python tools/benchmark_realtime.py --model xlsr_tiny --steps 4,6,8,10,20

# Production validation (20 runs)
python tools/benchmark_realtime.py --model xlsr_tiny --steps 6 --runs 20
```

### 2. Quick Benchmark (`quick_benchmark.py`)

**Purpose**: Rapid feasibility check

**Features**:
- ✅ Fast testing (3 runs by default)
- ✅ Single model/steps focus
- ✅ Immediate feasibility verdict
- ✅ Same accuracy as full benchmark

**Usage**:
```bash
# Quick check
python tools/quick_benchmark.py --model xlsr_tiny --steps 6
```

### 3. Visualization Tool (`visualize_benchmark.py`)

**Purpose**: Analyze and visualize benchmark results

**Features**:
- ✅ Feasibility heatmap (model × steps)
- ✅ Latency comparison bar charts
- ✅ Component breakdown stacked bars
- ✅ RTF comparison line charts
- ✅ Latency distribution histograms
- ✅ Text summary report

**Usage**:
```bash
python tools/visualize_benchmark.py benchmark_results.json
```

## Key Design Decisions

### 1. Feasibility Criteria

A configuration is **feasible** if:
- **Mean latency < 80% of block time** (20% headroom)
- **P99 latency < 100% of block time** (worst case fits)
- **RTF < 0.8**

**Rationale**:
- 80% mean threshold accounts for system variability (GC, background processes)
- P99 threshold ensures 99% of blocks process in time
- RTF < 0.8 provides safety margin for production use

### 2. Per-Component Timing

We measure latency for each pipeline component:
```
Total Latency = VAD + Resample + Content Encoder + Style Encoder 
               + Length Regulator + DiT Inference + Vocoder + Post-process
```

**Why**: Identifies bottlenecks for targeted optimization

### 3. Multiple Runs with Statistics

Default: **5 runs** per configuration (full benchmark), **3 runs** (quick)

**Computed metrics**:
- Mean, Std, Min, Max
- P50 (median), P90, P99 (percentiles)
- RTF (Real-Time Factor = latency / block_time)

**Why**: Single measurements are unreliable; statistics capture variability

### 4. System Detection

Auto-detects:
- Apple Silicon chip model (M1/M2/M3/M4)
- GPU core count (physical and logical)
- Unified memory (total and available)
- PyTorch device (MPS/CUDA)

**Why**: Performance varies significantly across hardware

### 5. JSON Output

Structured results enable:
- Programmatic analysis
- Comparison across runs
- Integration with CI/CD
- Historical tracking

## Benchmark Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. System Detection                                         │
│    - Detect chip, GPU, memory, device                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Model Loading                                            │
│    - Load config YAML                                        │
│    - Load checkpoint from HF or local                        │
│    - Initialize components (encoder, DiT, vocoder)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Warmup Run                                               │
│    - Single inference to warm caches                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Benchmark Runs (N iterations)                            │
│    For each run:                                            │
│    - Time VAD                                                │
│    - Time content encoder                                    │
│    - Time length regulator                                   │
│    - Time DiT inference                                      │
│    - Time vocoder                                            │
│    - Record total latency                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Statistics Computation                                   │
│    - Mean, std, percentiles                                  │
│    - RTF calculation                                         │
│    - Feasibility determination                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Output                                                   │
│    - Console summary                                         │
│    - JSON file                                               │
│    - (Optional) Visualizations                               │
└─────────────────────────────────────────────────────────────┘
```

## Validation of Original Claim

### How to Validate

1. **Run benchmark on target hardware**:
   ```bash
   python tools/benchmark_realtime.py --steps 4,6,8,10
   ```

2. **Check feasibility results**:
   - If `xlsr_tiny` with 6-8 steps is feasible → Claim supported
   - If `whisper_small` with any steps is feasible → Claim refuted
   - If no configs feasible → Hardware limitation

3. **Analyze component latencies**:
   - If DiT > 50% of total → DiT is bottleneck
   - If content encoder > 50% → Encoder is bottleneck
   - If vocoder > 30% → Vocoder is bottleneck

### Expected Results (M4 Max 48GB)

Based on theoretical analysis:

| Model | Steps | Expected Latency | Feasible? |
|-------|-------|-----------------|-----------|
| xlsr_tiny | 4 | ~80ms | ✅ Yes |
| xlsr_tiny | 6 | ~120ms | ✅ Yes |
| xlsr_tiny | 8 | ~160ms | ⚠️ Marginal |
| xlsr_tiny | 10 | ~200ms | ❌ No |
| whisper_small | 4 | ~250ms | ❌ No |
| whisper_small | 6 | ~350ms | ❌ No |

**Note**: These are estimates; actual benchmark will provide ground truth.

## Performance Optimization Insights

### From Component Timing

If benchmark reveals:

1. **Content encoder bottleneck** (>50ms):
   - Solution: Use XLSR instead of Whisper
   - Trade-off: Slightly lower quality

2. **DiT bottleneck** (>100ms):
   - Solution: Reduce diffusion steps
   - Trade-off: Lower audio quality

3. **Vocoder bottleneck** (>30ms):
   - Solution: Use HiFi-GAN instead of BigVGAN
   - Trade-off: Different audio characteristics

### Hardware-Specific Recommendations

**M1/M2 (8-10 GPU cores)**:
- Use `xlsr_tiny` with 4-6 steps
- Block time: 0.25s+
- Expected latency: 150-200ms

**M3/M4 Pro (14-20 GPU cores)**:
- Use `xlsr_tiny` with 6-8 steps
- Block time: 0.20s
- Expected latency: 120-160ms

**M4 Max (40 GPU cores)**:
- Use `xlsr_tiny` with 8-10 steps
- Block time: 0.20s
- May support `whisper_small` with 4 steps
- Expected latency: 100-150ms

## Testing and Validation

### Unit Tests (`test_benchmark.py`)

Tests cover:
- ✅ Module imports
- ✅ System detection
- ✅ Model configuration structure
- ✅ Result dataclass structure
- ✅ Feasibility logic
- ✅ Statistics computation
- ✅ RTF calculation

**Run tests**:
```bash
python tools/test_benchmark.py
```

### Integration Testing

1. **Quick test** (5 minutes):
   ```bash
   python tools/quick_benchmark.py --model xlsr_tiny --steps 6
   ```

2. **Full validation** (20 minutes):
   ```bash
   python tools/benchmark_realtime.py --steps 4,6,8,10 --runs 10
   ```

3. **Production validation** (30 minutes):
   ```bash
   python tools/benchmark_realtime.py --model xlsr_tiny --steps 6 --runs 20
   ```

## Output Examples

### Console Output

```
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
```

### JSON Output

```json
{
  "model_name": "xlsr_tiny",
  "diffusion_steps": 6,
  "mean_latency_ms": 137.0,
  "p99_latency_ms": 148.0,
  "rtf": 0.685,
  "feasible": true,
  "component_latencies": {
    "content_encoder": 45.3,
    "dit": 58.4,
    "vocoder": 22.0
  }
}
```

### Visualization Output

- `feasibility_heatmap.png`: Model × Steps feasibility matrix
- `latency_comparison.png`: Bar chart of latencies
- `component_breakdown.png`: Stacked bar of components
- `rtf_comparison.png`: RTF vs steps line chart
- `SUMMARY.md`: Text report

## Future Enhancements

### Phase 2 (Post-Validation)

1. **Memory profiling**:
   - Track peak memory usage
   - Identify memory leaks
   - Optimize memory allocation

2. **Quality metrics**:
   - PESQ (Perceptual Speech Quality)
   - STOI (Short-Time Objective Intelligibility)
   - Speaker similarity (embedding cosine similarity)

3. **Thermal monitoring**:
   - Track GPU temperature
   - Detect thermal throttling
   - Adjust benchmarks accordingly

4. **Batch processing**:
   - Benchmark multiple audio files
   - Compute average across speakers
   - Identify edge cases

5. **CI/CD integration**:
   - Automated benchmarking on PR
   - Performance regression detection
   - Hardware-specific test matrices

## Conclusion

This benchmark suite provides:

1. **Empirical validation** of the "25M only" real-time claim
2. **Hardware-specific recommendations** for users
3. **Bottleneck identification** for optimization
4. **Production-ready validation** with statistical rigor
5. **Extensible framework** for future enhancements

**Next steps**:
1. Run benchmarks on target hardware
2. Validate/refute original claim
3. Document hardware-specific recommendations
4. Integrate into deployment guide

## Files Created

```
tools/
├── benchmark_realtime.py      # Full benchmark suite (35KB)
├── quick_benchmark.py         # Quick feasibility checker (4KB)
├── visualize_benchmark.py     # Results visualization (13KB)
├── test_benchmark.py          # Unit tests (9KB)
├── requirements_benchmark.txt # Additional dependencies
├── README_BENCHMARK.md        # User documentation (10KB)
└── BENCHMARK_IMPLEMENTATION.md # This file
```

**Total**: ~71KB of production-ready benchmarking tools

## References

- [FINETUNING_AND_REALTIME_OPTIMIZATION.md](../docs/FINETUNING_AND_REALTIME_OPTIMIZATION.md)
- [Seed-VC README](https://github.com/Plachtaa/Seed-VC)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

---

**Implementation Date**: March 30, 2026
**Version**: 1.0.0
**Status**: Ready for testing
