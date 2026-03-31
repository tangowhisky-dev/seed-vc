#!/bin/bash
#
# Seed-VC Benchmark Interactive Launcher
#
# Auto-detects available models and audio files, then launches benchmarks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Seed-VC Benchmark Interactive Launcher"
echo "========================================"
echo ""

# Check if in correct directory
if [ ! -f "$SCRIPT_DIR/detect_models.py" ]; then
    echo "❌ Error: Script not found"
    exit 1
fi

# Detect available models
echo "🔍 Detecting available models..."
MODELS_JSON=$(python "$SCRIPT_DIR/detect_models.py" --output /tmp/seed_vc_models.json 2>/dev/null)
echo "$MODELS_JSON"

# Check if any models found
if ! python "$SCRIPT_DIR/detect_models.py" --output /tmp/seed_vc_models.json >/dev/null 2>&1; then
    echo "❌ No models detected. Please download models first."
    echo ""
    echo "To download models:"
    echo "  1. Run: python real-time-gui.py (auto-downloads xlsr_tiny)"
    echo "  2. Or: python tools/detect_models.py (to see what's available)"
    exit 1
fi

# Find available audio files
echo ""
echo "🔍 Finding audio files..."

# Find source audio
SOURCE_AUDIO=$(find "$ROOT_DIR/examples/source" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) 2>/dev/null | head -1)
if [ -z "$SOURCE_AUDIO" ]; then
    echo "⚠️  No source audio files found in examples/source/"
    SOURCE_AUDIO=""
else
    echo "✅ Source audio: $SOURCE_AUDIO"
fi

# Find reference audio
REF_AUDIO=$(find "$ROOT_DIR/examples/reference" -maxdepth 1 -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) 2>/dev/null | head -1)
if [ -z "$REF_AUDIO" ]; then
    echo "⚠️  No reference audio files found in examples/reference/"
    REF_AUDIO=""
else
    echo "✅ Reference audio: $REF_AUDIO"
fi

# Check if we have audio files
if [ -z "$SOURCE_AUDIO" ] || [ -z "$REF_AUDIO" ]; then
    echo ""
    echo "❌ Missing audio files. Please add:"
    echo "   - Source audio: examples/source/*.wav"
    echo "   - Reference audio: examples/reference/*.wav"
    exit 1
fi

echo ""
echo "========================================"
echo "Select Models to Benchmark"
echo "========================================"
echo ""

# Parse ready models from JSON
READY_MODELS=$(python3 -c "
import json
with open('/tmp/seed_vc_models.json') as f:
    data = json.load(f)
    ready = [m['name'] for m in data['models'] if m['ready_for_benchmark']]
    for m in ready:
        print(m)
" 2>/dev/null)

if [ -z "$READY_MODELS" ]; then
    echo "❌ No models ready for benchmark."
    echo "Some models may be missing dependencies."
    exit 1
fi

# Display models with checkboxes
echo "Available models (press SPACE to select, ENTER to confirm):"
echo ""

# Create a simple selection using Python
python3 << 'PYTHON_SCRIPT'
import json
import sys

with open('/tmp/seed_vc_models.json') as f:
    data = json.load(f)

ready_models = [m for m in data['models'] if m['ready_for_benchmark']]

if not ready_models:
    print("")
    sys.exit(1)

# Print models
print("Select models to benchmark (enter numbers separated by commas, or 'all'):")
print("")
for i, model in enumerate(ready_models, 1):
    desc = model.get('description', '')[:60]
    print(f"  [{i}] {model['name']:20s} - {desc}")

print("")
PYTHON_SCRIPT

# Use bash read for input
echo -n "Enter model numbers (e.g., 1,2,3 or 'all'): "
read -r MODEL_CHOICE

if [ -z "$MODEL_CHOICE" ]; then
    MODEL_CHOICE="all"
fi

# Parse selection using Python
SELECTED_MODELS=$(python3 << PYTHON_SCRIPT
import json

with open('/tmp/seed_vc_models.json') as f:
    data = json.load(f)

ready_models = [m for m in data['models'] if m['ready_for_benchmark']]
choice = "${MODEL_CHOICE}"

if choice.lower() == 'all':
    selected = [m['name'] for m in ready_models]
else:
    # Parse numbers
    indices = [int(x.strip()) for x in choice.split(',') if x.strip().isdigit()]
    selected = [ready_models[i-1]['name'] for i in indices if 1 <= i <= len(ready_models)]

# Output as space-separated list
print(' '.join(selected))
PYTHON_SCRIPT
)

if [ -z "$SELECTED_MODELS" ]; then
    echo "❌ No models selected"
    exit 1
fi

echo ""
echo "Selected models: $SELECTED_MODELS"
echo ""
echo "========================================"
echo "Select Benchmark Type"
echo "========================================"
echo ""
echo "  [1] Quick check (3 runs, ~1-2 min per model)"
echo "  [2] Full benchmark (5 runs, ~3-5 min per model)"
echo "  [3] Production validation (20 runs, ~10-15 min per model)"
echo ""
echo -n "Enter choice [1-3] (default: 1): "
read -r BENCHMARK_TYPE
BENCHMARK_TYPE=${BENCHMARK_TYPE:-1}

# Set runs based on type
case $BENCHMARK_TYPE in
    1) RUNS=3; BENCH_SCRIPT="quick_benchmark.py" ;;
    2) RUNS=5; BENCH_SCRIPT="benchmark_realtime.py" ;;
    3) RUNS=20; BENCH_SCRIPT="benchmark_realtime.py" ;;
    *) RUNS=3; BENCH_SCRIPT="quick_benchmark.py" ;;
esac

echo ""
echo "========================================"
echo "Select Diffusion Steps"
echo "========================================"
echo ""
echo "  [1] Quick: 4,6,8"
echo "  [2] Standard: 4,6,8,10"
echo "  [3] Comprehensive: 4,6,8,10,15,20"
echo "  [4] Custom (enter your own)"
echo ""
echo -n "Enter choice [1-4] (default: 2): "
read -r STEPS_CHOICE
STEPS_CHOICE=${STEPS_CHOICE:-2}

case $STEPS_CHOICE in
    1) STEPS="4,6,8" ;;
    2) STEPS="4,6,8,10" ;;
    3) STEPS="4,6,8,10,15,20" ;;
    4) 
        echo -n "Enter steps (comma-separated, e.g., 4,6,8,10): "
        read -r STEPS
        STEPS=${STEPS:-"4,6,8,10"}
        ;;
    *) STEPS="4,6,8,10" ;;
esac

echo ""
echo "Selected steps: $STEPS"
echo ""
echo "========================================"
echo "Configuration Summary"
echo "========================================"
echo ""
echo "Models: $SELECTED_MODELS"
echo "Benchmark type: $BENCHMARK_TYPE ($RUNS runs)"
echo "Diffusion steps: $STEPS"
echo "Source audio: $SOURCE_AUDIO"
echo "Reference audio: $REF_AUDIO"
echo ""
echo -n "Start benchmark? (y/n) [Y]: "
read -r CONFIRM
CONFIRM=${CONFIRM:-y}

if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "========================================"
echo "Running Benchmark"
echo "========================================"
echo ""

# Run benchmark for each selected model
for MODEL in $SELECTED_MODELS; do
    echo ""
    echo "----------------------------------------"
    echo "Benchmarking: $MODEL"
    echo "----------------------------------------"
    echo ""
    
    if [ "$BENCH_SCRIPT" = "quick_benchmark.py" ]; then
        python "$SCRIPT_DIR/quick_benchmark.py" \
            --model "$MODEL" \
            --steps "$STEPS" \
            --runs "$RUNS" \
            --test-audio "$SOURCE_AUDIO" \
            --reference-audio "$REF_AUDIO"
    else
        python "$SCRIPT_DIR/benchmark_realtime.py" \
            --model "$MODEL" \
            --steps "$STEPS" \
            --runs "$RUNS" \
            --test-audio "$SOURCE_AUDIO" \
            --reference-audio "$REF_AUDIO"
    fi
    
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "⚠️  Benchmark for $MODEL completed with warnings"
    fi
done

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
ls -1 benchmark_results_*.json quick_benchmark_*.json 2>/dev/null | tail -5

echo ""
echo "To visualize results:"
echo "  pip install matplotlib seaborn"
echo "  python $SCRIPT_DIR/visualize_benchmark.py benchmark_results_*.json"
echo ""
