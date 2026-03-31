# Seed-VC Benchmark Suite - Auto-Detection Edition

## 🎯 Overview

This benchmark suite **automatically detects** available models in your Seed-VC installation and validates real-time feasibility with minimal user input.

## ✨ Key Features

1. **Auto-Detection**: Scans checkpoints directory for available models
2. **Smart Fallback**: Downloads models from HuggingFace if not found locally
3. **Interactive Launch**: `examples.sh` provides menu-driven interface
4. **Comprehensive Testing**: Per-component timing, statistical analysis
5. **Visualization**: Generate charts and reports from results

## 📁 File Structure

```
tools/
├── benchmark_realtime.py          # Full benchmark (auto-detects models)
├── quick_benchmark.py             # Quick feasibility check
├── detect_models.py               # Model auto-detection utility
├── visualize_benchmark.py         # Results visualization
├── test_benchmark.py              # Unit tests
├── examples.sh                    # Interactive launcher (NEW!)
├── README_BENCHMARK.md            # Full documentation
├── BENCHMARK_IMPLEMENTATION.md    # Technical details
├── IMPLEMENTATION_SUMMARY.md      # Quick reference
└── AUTO_DETECTION_README.md       # This file
```

## 🚀 Quick Start

### Option 1: Interactive Mode (Recommended)

```bash
cd /path/to/seed-vc
./tools/examples.sh
```

This will:
1. ✅ Auto-detect available models
2. ✅ Find audio files in `examples/` directory
3. ✅ Show interactive menu to select models
4. ✅ Run benchmark with your chosen settings
5. ✅ Save results to JSON

### Option 2: Command Line

```bash
# Quick check (auto-detects models)
python tools/quick_benchmark.py

# Full benchmark with all detected models
python tools/benchmark_realtime.py

# Specific model
python tools/benchmark_realtime.py --model xlsr_tiny --steps 4,6,8,10
```

### Option 3: Model Detection Only

```bash
# See what models are available
python tools/detect_models.py

# Save to JSON
python tools/detect_models.py --output available_models.json
```

## 🔍 How Model Auto-Detection Works

### Detection Process

1. **Scan Checkpoints Directory**
   ```
   ./checkpoints/
   ├── models--Plachta--Seed-VC/
   │   └── blobs/
   │       ├── 73ae186... (V2 CFM)
   │       └── 42e2afa... (V2 AR)
   └── models--funasr--campplus/
       └── blobs/
           └── campplus_cn_common.bin
   ```

2. **Check HuggingFace Cache**
   ```
   ~/.cache/huggingface/hub/
   ├── models--facebook--wav2vec2-xls-r-300m/
   ├── models--openai--whisper-small/
   ├── models--nvidia--bigvgan_v2_22khz_80band_256x/
   └── models--FunAudioLLM--CosyVoice-300M/
   ```

3. **Match Against Known Models**
   - `DiT_uvit_tat_xlsr_ema.pth` → `xlsr_tiny`
   - `DiT_seed_v2_uvit_whisper_small...pth` → `whisper_small`
   - `DiT_seed_v2_uvit_whisper_base...pth` → `whisper_base`
   - `cfm_small.pth` → `v2_cfm_small`
   - `ar_base.pth` → `v2_ar_base`

4. **Check Dependencies**
   - XLSR model requires: `facebook/wav2vec2-xls-r-300m`, HiFi-GAN
   - Whisper models require: `openai/whisper-small`, BigVGAN
   - All models require: `campplus_cn_common.bin`

5. **Mark as Ready**
   - ✅ All dependencies present → Ready for benchmark
   - ⚠️ Missing dependencies → Will download on first use
   - ❌ V2 models → Not supported (yet)

### Detection Output

```
====================================================================================================
DETECTED MODELS
====================================================================================================
Model Name               Size     Params Vocoder      Tokenizer  Ready    Status
----------------------------------------------------------------------------------------------------
v2_cfm_small           337.0MB        67M bigvgan      astral     ⚠️ V2 (not benchmarked)
v2_ar_base             342.2MB        90M bigvgan      astral     ⚠️ V2 (not benchmarked)
====================================================================================================

Total models found: 2
Ready for benchmark: 0

⚠️  No models ready for benchmark.

To download models:
  1. Run: python real-time-gui.py (will auto-download xlsr_tiny on first use)
  2. Models will be saved to: ./checkpoints/
  3. Re-run: python tools/detect_models.py
```

## 📦 Downloading Models

### Method 1: Auto-Download via real-time-gui.py

```bash
# First run will auto-download xlsr_tiny model
python real-time-gui.py
```

**What happens**:
1. Script checks for `DiT_uvit_tat_xlsr_ema.pth`
2. If not found, downloads from `https://huggingface.co/Plachta/Seed-VC`
3. Downloads dependencies (XLSR, HiFi-GAN, CAMPPlus)
4. Saves to `./checkpoints/`

### Method 2: Manual Download

```bash
# Download specific model
huggingface-cli download Plachta/Seed-VC DiT_uvit_tat_xlsr_ema.pth --local-dir ./checkpoints

# Or use wget/curl
wget https://huggingface.co/Plachta/Seed-VC/resolve/main/DiT_uvit_tat_xlsr_ema.pth \
  -O ./checkpoints/DiT_uvit_tat_xlsr_ema.pth
```

### Method 3: Use detect_models.py Helper

```bash
# See what's available and what's missing
python tools/detect_models.py --verbose

# Follow the instructions to download missing models
```

## 🎮 Interactive Launcher (examples.sh)

### Features

1. **Auto-detect models** from checkpoints directory
2. **Find audio files** in `examples/source/` and `examples/reference/`
3. **Interactive model selection** with checkboxes
4. **Benchmark type selection** (quick/full/production)
5. **Diffusion steps selection** (preset or custom)
6. **Configuration summary** before running
7. **Batch processing** for multiple models

### Usage

```bash
cd /path/to/seed-vc
./tools/examples.sh
```

### Sample Interaction

```
========================================
Seed-VC Benchmark Interactive Launcher
========================================

🔍 Detecting available models...
✅ Found 1 model(s)

🔍 Finding audio files...
✅ Source audio: examples/source/source_s1.wav
✅ Reference audio: examples/reference/s1p1.wav

========================================
Select Models to Benchmark
========================================

Select models to benchmark (space to toggle, enter to confirm):

  [1] xlsr_tiny            - 25M DiT + XLSR-large (300M frozen) + HiFi-GAN

Enter model numbers separated by commas (e.g., 1,2 or 'all' for all): 1

Selected models: xlsr_tiny

========================================
Select Benchmark Type
========================================

  [1] Quick check (3 runs, ~1-2 min per model)
  [2] Full benchmark (5 runs, ~3-5 min per model)
  [3] Production validation (20 runs, ~10-15 min per model)

Enter choice [1-3] (default: 1): 1

========================================
Select Diffusion Steps
========================================

  [1] Quick: 4,6,8
  [2] Standard: 4,6,8,10
  [3] Comprehensive: 4,6,8,10,15,20
  [4] Custom (enter your own)

Enter choice [1-4] (default: 2): 2

Selected steps: 4,6,8,10

========================================
Configuration Summary
========================================

Models: xlsr_tiny
Benchmark type: 1 (3 runs)
Diffusion steps: 4,6,8,10
Source audio: examples/source/source_s1.wav
Reference audio: examples/reference/s1p1.wav

Start benchmark? (y/n) [Y]: y

========================================
Running Benchmark
========================================
```

## 📊 Available Audio Files

### Source Audio (Input)

Located in `examples/source/`:
- `source_s1.wav` through `source_s4.wav`
- `jay_0.wav`, `yae_0.wav`, `glados_0.wav`
- `TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav`
- `Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav`

### Reference Audio (Target Voice)

Located in `examples/reference/`:
- `s1p1.wav`, `s1p2.wav` (Speaker 1)
- `s2p1.wav`, `s2p2.wav` (Speaker 2)
- `s3p1.wav`, `s3p2.wav` (Speaker 3)
- `s4p1.wav`, `s4p2.wav` (Speaker 4)
- `trump_0.wav`, `teio_0.wav`, `dingzhen_0.wav`

## 🧪 Benchmark Workflows

### Workflow 1: Quick Validation (5 minutes)

```bash
# See what models are available
python tools/detect_models.py

# Run quick benchmark on first available model
python tools/quick_benchmark.py --steps 6
```

### Workflow 2: Comprehensive Testing (20 minutes)

```bash
# Interactive mode
./tools/examples.sh

# Or command line
python tools/benchmark_realtime.py --steps 4,6,8,10 --runs 5
```

### Workflow 3: Production Validation (30 minutes)

```bash
# High-confidence validation with 20 runs
python tools/benchmark_realtime.py --model xlsr_tiny --steps 6 --runs 20

# Visualize results
pip install matplotlib seaborn
python tools/visualize_benchmark.py benchmark_results_*.json
```

## 📈 Result Interpretation

### Feasible Configuration

```
✅ xlsr_tiny (6 steps)
------------------------------------------------------------
Block time: 0.20s (200ms)
Latency - Mean: 137ms, P50: 135ms, P90: 142ms, P99: 148ms
RTF: 0.685 (Real-Time Factor)
Feasibility - Mean < 80% block: ✅, P99 < 100% block: ✅
```

**Interpretation**:
- ✅ Mean latency (137ms) < 160ms (80% of 200ms block)
- ✅ P99 latency (148ms) < 200ms (100% of block)
- ✅ RTF (0.685) < 0.8
- **Verdict**: Feasible for real-time use

### Not Feasible Configuration

```
❌ whisper_small (6 steps)
------------------------------------------------------------
Block time: 0.20s (200ms)
Latency - Mean: 352ms, P50: 350ms, P90: 360ms, P99: 375ms
RTF: 1.760 (Real-Time Factor)
Feasibility - Mean < 80% block: ❌, P99 < 100% block: ❌
```

**Interpretation**:
- ❌ Mean latency (352ms) > 160ms
- ❌ P99 latency (375ms) > 200ms
- ❌ RTF (1.760) > 0.8
- **Verdict**: NOT feasible for real-time use

## 🔧 Troubleshooting

### "No models found"

```bash
# Check what's in checkpoints
python tools/detect_models.py --verbose

# Download models
python real-time-gui.py  # Auto-downloads on first run
```

### "Audio files not found"

```bash
# Check examples directory
ls examples/source/
ls examples/reference/

# Add audio files if missing
# Place source audio in: examples/source/*.wav
# Place reference audio in: examples/reference/*.wav
```

### "Missing dependencies"

```bash
# See what's missing
python tools/detect_models.py --verbose

# Run real-time-gui.py to auto-download
python real-time-gui.py
```

### "Import error"

```bash
# Install dependencies
pip install -r requirements.txt
pip install psutil  # For system detection
pip install matplotlib seaborn  # For visualization (optional)
```

## 📚 Documentation

- **Full Guide**: `tools/README_BENCHMARK.md`
- **Technical Details**: `tools/BENCHMARK_IMPLEMENTATION.md`
- **Quick Reference**: `tools/IMPLEMENTATION_SUMMARY.md`
- **This Guide**: `tools/AUTO_DETECTION_README.md`

## 🎯 Best Practices

1. **Start with detection**: Always run `python tools/detect_models.py` first
2. **Use interactive mode**: `./tools/examples.sh` for easiest experience
3. **Quick test first**: Use `quick_benchmark.py` before full benchmark
4. **Save results**: JSON files are useful for comparison
5. **Visualize**: Generate charts to understand bottlenecks

## 🔄 Updating Models

```bash
# Remove old checkpoints
rm -rf ./checkpoints/models--Plachta--Seed-VC/blobs/*

# Re-download latest models
python real-time-gui.py

# Verify detection
python tools/detect_models.py
```

## 📝 Notes

- **V2 Models**: Currently not supported for benchmarking (require special handling)
- **Auto-download**: First benchmark run will download missing dependencies
- **Cache location**: Models cached in `./checkpoints/` and `~/.cache/huggingface/hub/`
- **Disk space**: Expect ~2-3GB for all models and dependencies

---

**Last Updated**: March 30, 2026
**Version**: 2.0 (with auto-detection)
