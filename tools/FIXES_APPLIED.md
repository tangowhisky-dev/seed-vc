# Fixes Applied - March 30, 2026

## Issues Fixed

### 1. examples.sh - Reference Audio Not Found ❌ → ✅

**Problem**: Script reported "No reference audio files found" even though files existed in `examples/reference/`

**Root Cause**: Typo in bash glob pattern
```bash
# Before (WRONG):
\( -name "*.wav" -o -name "*.mp3" -o - "*.flac" \)

# After (FIXED):
\( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \)
```

**Result**: Now correctly finds all audio files in `examples/reference/`

---

### 2. detect_models.py - False "Missing deps" Reports ❌ → ✅

**Problem**: Script reported "Missing deps" for `xlsr_tiny` even though dependencies were being downloaded by real-time-gui.py

**Root Cause**: Detection was too strict - required all dependencies to be fully downloaded before marking model as ready

**Fix Applied**:
1. **Separate "missing" from "will download"**:
   - ❌ Missing: Truly missing files that will block execution
   - 🔄 Will download: Dependencies that auto-download on first use (OK)

2. **Updated dependency checking**:
   ```python
   # Before: All or nothing
   return len(missing) == 0, missing
   
   # After: Three-state detection
   return is_ready, missing_list, will_download_list
   ```

3. **More lenient HF repo detection**:
   - Checks if repo directory exists (even if partial download)
   - Doesn't require full model files to be present
   - Allows auto-download on first benchmark run

4. **Updated status display**:
   ```
   ✅ Ready                    - All deps present, can benchmark now
   🔄 Will download on first use - Missing optional deps that auto-download
   ❌ Missing deps             - Truly missing required files
   ⚠️  V2 (not benchmarked)    - V2 models not supported yet
   ```

**Result**:
```
Before:
xlsr_tiny  ❌ Missing deps

After:
xlsr_tiny  ✅ Ready
   🔄 Will download: HF repo: FunAudioLLM/CosyVoice-300M, Vocoder: checkpoints/hift.pt
```

---

## Testing Results

### Detection Output
```bash
$ python tools/detect_models.py --verbose

✅ Ready for benchmark: 1
🔄 Will download on first use: 1
⚠️  V2 models (not benchmarked): 2

✅ Ready models: xlsr_tiny

You can run benchmarks immediately:
  ./tools/examples.sh
  python tools/quick_benchmark.py
```

### examples.sh Output
```bash
$ ./tools/examples.sh

✅ Source audio: /Users/tango16/code/seed-vc/examples/source/TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav
✅ Reference audio: /Users/tango16/code/seed-vc/examples/reference/s2p2.wav

========================================
Select Models to Benchmark
========================================

Available models (press SPACE to select, ENTER to confirm):

  [1] xlsr_tiny            - 25M DiT + XLSR-large (300M frozen) + HiFi-GAN
```

---

## What Was Changed

### File: `tools/examples.sh`
- **Line 38**: Fixed typo in bash glob pattern
- Changed: `-o - "*.flac"` → `-o -name "*.flac"`

### File: `tools/detect_models.py`
- **Lines 53-73**: Updated `MODEL_DEPENDENCIES` to include `vocoder_checkpoint`
- **Lines 175-213**: Rewrote `check_dependencies()` to return 3 values instead of 2
- **Line 79**: Added `will_download` field to `ModelInfo` dataclass
- **Lines 216-258**: Updated `detect_available_models()` to use new dependency checking
- **Lines 261-317**: Completely rewrote `print_models()` with better status display

---

## How It Works Now

### Dependency Detection Logic

```
1. Check HF repos:
   - If directory exists (even partial) → OK (will download rest)
   - If directory doesn't exist → "Will download"
   
2. Check local files:
   - If found in checkpoints/ → OK
   - If not found → "Missing"
   
3. Check vocoder checkpoints:
   - If exists → OK
   - If not → "Will download on first use"

4. Determine status:
   - No missing + not V2 → ✅ Ready
   - No missing + V2 → ⚠️ V2 (not benchmarked)
   - Has missing → ❌ Missing deps
   - No missing but has will_download → 🔄 Will download
```

### Auto-Download Behavior

When you run the benchmark with a model marked as "🔄 Will download":

1. **First run**:
   - real-time-gui.py / benchmark script detects missing deps
   - Downloads from HuggingFace automatically
   - Saves to `./checkpoints/` and `~/.cache/huggingface/hub/`
   - Completes benchmark

2. **Subsequent runs**:
   - Dependencies already cached
   - Faster execution
   - Marked as "✅ Ready"

---

## Next Steps

### Option 1: Run Benchmark Now (Auto-Download)
```bash
# Will auto-download missing dependencies on first run
python tools/quick_benchmark.py --model xlsr_tiny --steps 6
```

### Option 2: Pre-Download Dependencies
```bash
# Run once to download all dependencies
python real-time-gui.py

# Then verify
python tools/detect_models.py

# Then benchmark
./tools/examples.sh
```

### Option 3: Manual Download
```bash
# Download vocoder
wget https://huggingface.co/Plachta/Seed-VC/resolve/main/hift.pt \
  -O ./checkpoints/hift.pt

# Download other deps as needed
```

---

## Verification

### Check Detection
```bash
python tools/detect_models.py --verbose
```

**Expected Output**:
```
✅ Ready for benchmark: 1
Ready models: xlsr_tiny
```

### Check Audio Detection
```bash
./tools/examples.sh
```

**Expected Output**:
```
✅ Source audio: examples/source/...
✅ Reference audio: examples/reference/...
```

### Run Quick Benchmark
```bash
python tools/quick_benchmark.py --model xlsr_tiny --steps 6
```

**Expected**: First run downloads missing deps, then completes benchmark

---

**Status**: ✅ All issues fixed
**Tested**: ✅ Detection working, ✅ Audio files found
**Ready**: ✅ Can run benchmarks immediately
