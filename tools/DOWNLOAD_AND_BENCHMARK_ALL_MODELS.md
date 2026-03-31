# Download and Benchmark All Models Guide

## Available Models

Based on the Seed-VC README, there are **4 main models** (5 variants):

| Version | Name | Params | Purpose | Sampling Rate | Content Encoder | Vocoder |
|---------|------|--------|---------|---------------|-----------------|---------|
| v1.0 | **xlsr_tiny** | 25M | Voice Conversion (real-time) | 22.05kHz | XLSR-large | HIFT |
| v1.0 | **whisper_small** | 98M | Voice Conversion (offline) | 22.05kHz | Whisper-small | BigVGAN |
| v1.0 | **whisper_base** | 200M | Singing Voice Conversion | 44.1kHz | Whisper-small | BigVGAN |
| v2.0 | **v2_cfm_small** | 67M | Voice & Accent Conversion | 22.05kHz | ASTRAL-Quantization | BigVGAN |
| v2.0 | **v2_ar_base** | 90M | Voice & Accent Conversion | 22.05kHz | ASTRAL-Quantization | BigVGAN |

**Total**: ~2GB of model files

---

## Quick Start: Download All Models

```bash
# Download all 5 models (~2GB, 10-30 minutes)
python tools/download_all_models.py --all
```

This will download:
- ✅ All 5 model checkpoints
- ✅ All dependencies (XLSR, Whisper, BigVGAN, etc.)
- ✅ All vocoders (HIFT, BigVGAN)

---

## Step-by-Step: Download Individual Models

### 1. Download xlsr_tiny (25M, real-time)

```bash
python tools/download_all_models.py --model xlsr_tiny
```

**What gets downloaded**:
- `DiT_uvit_tat_xlsr_ema.pth` (135MB)
- `facebook/wav2vec2-xls-r-300m` (1.2GB)
- `FunAudioLLM/CosyVoice-300M`
- `hift.pt` (vocoder)

**Best for**: Real-time voice conversion

---

### 2. Download whisper_small (98M, offline)

```bash
python tools/download_all_models.py --model whisper_small
```

**What gets downloaded**:
- `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` (342MB)
- `openai/whisper-small` (600MB)
- `nvidia/bigvgan_v2_22khz_80band_256x` (400MB)
- `bigvgan_generator.pt` (vocoder)

**Best for**: High-quality offline voice conversion

---

### 3. Download whisper_base (200M, SVC)

```bash
python tools/download_all_models.py --model whisper_base
```

**What gets downloaded**:
- `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` (850MB)
- `openai/whisper-small` (600MB)
- `nvidia/bigvgan_v2_22khz_80band_256x` (400MB)
- `bigvgan_generator.pt` (vocoder)

**Best for**: Singing voice conversion (44.1kHz, strong zero-shot)

---

### 4. Download v2_cfm_small (67M, accent conversion)

```bash
python tools/download_all_models.py --model v2_cfm_small
```

**What gets downloaded**:
- `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth` (337MB)
- `nvidia/bigvgan_v2_22khz_80band_256x` (400MB)
- `bigvgan_generator.pt` (vocoder)

**Best for**: Accent conversion, suppressing source speaker traits

**Note**: V2 models require special handling (not yet benchmarked)

---

### 5. Download v2_ar_base (90M, accent conversion)

```bash
python tools/download_all_models.py --model v2_ar_base
```

**What gets downloaded**:
- `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth` (342MB)
- `nvidia/bigvgan_v2_22khz_80band_256x` (400MB)
- `bigvgan_generator.pt` (vocoder)

**Best for**: Accent conversion, suppressing source speaker traits

**Note**: V2 models require special handling (not yet benchmarked)

---

## Verify Download

```bash
# Check what models are available
python tools/detect_models.py

# Expected output:
# ✅ Ready for benchmark: 3 (xlsr_tiny, whisper_small, whisper_base)
# ⚠️  V2 models (not benchmarked): 2 (v2_cfm_small, v2_ar_base)
```

---

## Benchmark All Models

### Option 1: Interactive Mode (Recommended)

```bash
./tools/examples.sh
```

This will:
1. Show all available models
2. Let you select which to benchmark
3. Choose benchmark type (quick/full/production)
4. Run benchmarks for selected models

### Option 2: Command Line

```bash
# Benchmark all V1 models (real-time and offline)
python tools/benchmark_realtime.py --model xlsr_tiny whisper_small whisper_base --steps 4,6,8,10

# Or benchmark each separately
python tools/benchmark_realtime.py --model xlsr_tiny --steps 4,6,8,10
python tools/benchmark_realtime.py --model whisper_small --steps 4,6,8,10
python tools/benchmark_realtime.py --model whisper_base --steps 4,6,8,10
```

### Option 3: Quick Check

```bash
# Quick benchmark for all models
python tools/quick_benchmark.py --model xlsr_tiny --steps 6
python tools/quick_benchmark.py --model whisper_small --steps 6
python tools/quick_benchmark.py --model whisper_base --steps 6
```

---

## Expected Results

### xlsr_tiny (25M)
- **Expected latency**: ~80-200ms (4-10 steps)
- **Real-time feasible**: ✅ Yes (with 4-8 steps)
- **Best use case**: Real-time voice conversion

### whisper_small (98M)
- **Expected latency**: ~250-400ms (4-10 steps)
- **Real-time feasible**: ❌ No (too slow for real-time)
- **Best use case**: Offline voice conversion

### whisper_base (200M)
- **Expected latency**: ~500-800ms (4-10 steps)
- **Real-time feasible**: ❌ No (much slower)
- **Best use case**: Singing voice conversion (quality over speed)

### v2_cfm_small (67M)
- **Benchmarking**: ⚠️ Not yet supported
- **Best use case**: Accent conversion

### v2_ar_base (90M)
- **Benchmarking**: ⚠️ Not yet supported
- **Best use case**: Accent conversion

---

## Disk Space Requirements

| Component | Size |
|-----------|------|
| Model checkpoints | ~2GB |
| XLSR-large | ~1.2GB |
| Whisper-small | ~600MB |
| BigVGAN | ~400MB |
| Other dependencies | ~500MB |
| **Total** | **~4.7GB** |

**Recommendation**: Ensure at least 10GB free space

---

## Troubleshooting

### "No models found"

```bash
# Download models first
python tools/download_all_models.py --all

# Or download individually
python tools/download_all_models.py --model xlsr_tiny
```

### "Download failed"

```bash
# Check internet connection
ping huggingface.co

# Try with HTTP proxy if needed
export HF_ENDPOINT=https://huggingface.co
python tools/download_all_models.py --model xlsr_tiny
```

### "Out of disk space"

```bash
# Check available space
df -h .

# Clean up HuggingFace cache if needed
rm -rf ~/.cache/huggingface/hub/models--*

# Or download only needed models
python tools/download_all_models.py --model xlsr_tiny
```

### "Import error"

```bash
# Install dependencies
pip install -r requirements.txt
pip install psutil  # For system detection
```

---

## Alternative: Manual Download

If the download script doesn't work, you can download manually:

```bash
# Using huggingface-cli
huggingface-cli download Plachta/Seed-VC DiT_uvit_tat_xlsr_ema.pth --local-dir ./checkpoints
huggingface-cli download Plachta/Seed-VC DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth --local-dir ./checkpoints
huggingface-cli download Plachta/Seed-VC DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth --local-dir ./checkpoints

# Using wget
wget https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth -O ./checkpoints/DiT_uvit_tat_xlsr_ema.pth
```

---

## Summary

**To download and test all models**:

```bash
# Step 1: Download all models
python tools/download_all_models.py --all

# Step 2: Verify detection
python tools/detect_models.py

# Step 3: Benchmark all V1 models
./tools/examples.sh

# Step 4: Visualize results
python tools/visualize_benchmark.py benchmark_results_*.json
```

**Total time**: ~30-60 minutes (download + benchmark)

---

**Status**: Ready to download and benchmark all models
**Recommended**: Start with `xlsr_tiny` (fastest, real-time capable)
