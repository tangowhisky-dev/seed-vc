# Fix: Interactive Input in examples.sh

## Issue

**Problem**: Script hung after showing "Select Models to Benchmark" prompt

**Symptoms**:
```
========================================
Select Models to Benchmark
========================================

Available models (press SPACE to select, ENTER to confirm):

(seedvc) tango16@tw-macbook seed-vc %  ← Script just quit/hung
```

**Root Cause**: Python's `input()` inside a heredoc was waiting for input but the prompt wasn't displaying correctly in the terminal.

## Fix Applied

### Before (Broken)
```bash
SELECTED_MODELS=$(python3 << 'PYTHON_SCRIPT'
    # ... code ...
    print("Enter model numbers:", end=" ")
    choice = input().strip()  # ← This was waiting but prompt not showing
PYTHON_SCRIPT
)
```

### After (Fixed)
```bash
# Step 1: Display models using Python
python3 << 'PYTHON_SCRIPT'
    # ... code ...
    print("Select models to benchmark:")
    for i, model in enumerate(ready_models, 1):
        print(f"  [{i}] {model['name']}")
PYTHON_SCRIPT

# Step 2: Get input using bash read
echo -n "Enter model numbers (e.g., 1,2,3 or 'all'): "
read -r MODEL_CHOICE

# Step 3: Parse selection using Python
SELECTED_MODELS=$(python3 << PYTHON_SCRIPT
    choice = "${MODEL_CHOICE}"  # ← Input from bash
    # ... parsing logic ...
PYTHON_SCRIPT
)
```

## How It Works Now

1. **Display models** (Python): Shows the list of available models
2. **Get input** (bash): Uses `read` to capture user input with visible prompt
3. **Parse selection** (Python): Converts input to model names

## Testing

```bash
./tools/examples.sh
```

**Expected flow**:
```
========================================
Select Models to Benchmark
========================================

Select models to benchmark (enter numbers separated by commas, or 'all'):

  [1] xlsr_tiny            - V1.0: 25M DiT + XLSR-large...
  [2] whisper_small        - V1.0: 98M DiT + Whisper-small...
  [3] whisper_base         - V1.0: 200M DiT + Whisper-small...

Enter model numbers (e.g., 1,2,3 or 'all'): all  ← User types here

Selected models: xlsr_tiny whisper_small whisper_base
```

## Files Modified

- `tools/examples.sh`:
  - Separated model display from input collection
  - Used bash `read` for interactive input
  - Simplified Python code to only handle display and parsing

## Status

✅ **Fixed**: Interactive input now works correctly
✅ **Tested**: Script proceeds through all prompts
✅ **Ready**: Can now run benchmarks interactively
