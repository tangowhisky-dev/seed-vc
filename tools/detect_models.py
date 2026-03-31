#!/usr/bin/env python3
"""
Model Auto-Detection for Seed-VC Benchmark

Scans checkpoints directory and HuggingFace cache to detect available models.
Returns a list of models with metadata (size, type, config).

Usage:
    python tools/detect_models.py [--output JSON_FILE]
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

# Checkpoints directory
CHECKPOINTS_DIR = Path("./checkpoints")
HF_CACHE_DIR = Path(os.path.expanduser("~/.cache/huggingface/hub"))

# Known model mappings
KNOWN_MODELS = {
    # V1 Models (real-time and offline VC)
    "DiT_uvit_tat_xlsr_ema.pth": {
        "name": "xlsr_tiny",
        "config": "config_dit_mel_seed_uvit_xlsr_tiny.yml",
        "description": "V1.0: 25M DiT + XLSR-large (300M frozen) + HIFT (real-time VC)",
        "estimated_params_m": 25,
        "vocoder": "hifigan",
        "tokenizer": "xlsr",
        "version": "v1.0",
        "purpose": "Voice Conversion (real-time)",
        "sampling_rate": 22050
    },
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth": {
        "name": "whisper_small",
        "config": "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        "description": "V1.0: 98M DiT + Whisper-small (39M) + BigVGAN (offline VC)",
        "estimated_params_m": 98,
        "vocoder": "bigvgan",
        "tokenizer": "whisper",
        "version": "v1.0",
        "purpose": "Voice Conversion (offline)",
        "sampling_rate": 22050
    },
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth": {
        "name": "whisper_base",
        "config": "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "description": "V1.0: 200M DiT + Whisper-small (39M) + BigVGAN (SVC, 44.1kHz)",
        "estimated_params_m": 200,
        "vocoder": "bigvgan",
        "tokenizer": "whisper",
        "version": "v1.0",
        "purpose": "Singing Voice Conversion",
        "sampling_rate": 44100
    },
    # V2 Models (CFM and AR)
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth": {
        "name": "v2_cfm_small",
        "config": None,  # V2 model, needs special handling
        "description": "V2.0: 67M CFM + ASTRAL-Quantization + BigVGAN (accent conversion)",
        "estimated_params_m": 67,
        "vocoder": "bigvgan",
        "tokenizer": "astral",
        "version": "v2.0",
        "purpose": "Voice & Accent Conversion",
        "sampling_rate": 22050,
        "v2_model": True,
        "model_type": "CFM"
    },
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth": {
        "name": "v2_ar_base",
        "config": None,  # V2 model, needs special handling
        "description": "V2.0: 90M AR + ASTRAL-Quantization + BigVGAN (accent conversion)",
        "estimated_params_m": 90,
        "vocoder": "bigvgan",
        "tokenizer": "astral",
        "version": "v2.0",
        "purpose": "Voice & Accent Conversion",
        "sampling_rate": 22050,
        "v2_model": True,
        "model_type": "AR"
    },
}

# Required dependencies for each model
MODEL_DEPENDENCIES = {
    "xlsr_tiny": {
        "hf_repos": ["facebook/wav2vec2-xls-r-300m", "FunAudioLLM/CosyVoice-300M"],
        "local_files": ["campplus_cn_common.bin"],
        "vocoder_checkpoint": "checkpoints/hift.pt"  # Will be downloaded on first use
    },
    "whisper_small": {
        "hf_repos": ["openai/whisper-small", "nvidia/bigvgan_v2_22khz_80band_256x"],
        "local_files": ["campplus_cn_common.bin"],
        "vocoder_checkpoint": "checkpoints/bigvgan_generator.pt"  # Will be downloaded on first use
    },
    "whisper_base": {
        "hf_repos": ["openai/whisper-small", "nvidia/bigvgan_v2_22khz_80band_256x"],
        "local_files": ["campplus_cn_common.bin"],
        "vocoder_checkpoint": "checkpoints/bigvgan_generator.pt"  # Will be downloaded on first use
    },
    "v2_cfm_small": {
        "hf_repos": ["nvidia/bigvgan_v2_22khz_80band_256x"],
        "local_files": ["campplus_cn_common.bin", "bsq32_light.pth"]
    },
    "v2_ar_base": {
        "hf_repos": ["nvidia/bigvgan_v2_22khz_80band_256x"],
        "local_files": ["campplus_cn_common.bin", "bsq32_light.pth"]
    }
}


@dataclass
class ModelInfo:
    """Information about a detected model."""
    name: str
    filename: str
    path: str
    size_mb: float
    description: str
    estimated_params_m: float
    config: Optional[str]
    vocoder: str
    tokenizer: str
    is_v2: bool
    dependencies_available: bool
    missing_dependencies: List[str]
    will_download: List[str]
    ready_for_benchmark: bool


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except:
        return 0.0


def find_checkpoint_files() -> Dict[str, Path]:
    """Find all checkpoint files in checkpoints directory."""
    checkpoints = {}
    
    if not CHECKPOINTS_DIR.exists():
        return checkpoints
    
    # Search in blobs directories
    for blob_dir in CHECKPOINTS_DIR.glob("*/blobs/*"):
        if blob_dir.is_file():
            # Try to identify the model from the blob
            checkpoints[blob_dir.name] = blob_dir
    
    # Search in snapshots directories (for named files)
    for snapshot_dir in CHECKPOINTS_DIR.glob("*/snapshots/*"):
        if snapshot_dir.is_dir():
            for file_path in snapshot_dir.iterdir():
                if file_path.is_file() and (file_path.suffix in ['.pth', '.pt', '.bin']):
                    checkpoints[file_path.name] = file_path
    
    # Also check HuggingFace repo structure
    for repo_dir in CHECKPOINTS_DIR.glob("models--*"):
        if repo_dir.is_dir():
            for file_path in repo_dir.rglob("*.pth"):
                if file_path.is_file():
                    checkpoints[file_path.name] = file_path
    
    return checkpoints


def find_hf_cache_models() -> Dict[str, Path]:
    """Find models in HuggingFace cache."""
    hf_models = {}
    
    if not HF_CACHE_DIR.exists():
        return hf_models
    
    # Check for known repos
    known_repos = [
        "models--facebook--wav2vec2-xls-r-300m",
        "models--openai--whisper-small",
        "models--nvidia--bigvgan_v2_22khz_80band_256x",
        "models--FunAudioLLM--CosyVoice-300M",
        "models--funasr--campplus"
    ]
    
    for repo in known_repos:
        repo_path = HF_CACHE_DIR / repo
        if repo_path.exists():
            # Find model files
            for file_path in repo_path.rglob("*.pth"):
                if file_path.is_file():
                    hf_models[f"{repo}/{file_path.name}"] = file_path
            for file_path in repo_path.rglob("*.pt"):
                if file_path.is_file():
                    hf_models[f"{repo}/{file_path.name}"] = file_path
            for file_path in repo_path.rglob("*.bin"):
                if file_path.is_file():
                    hf_models[f"{repo}/{file_path.name}"] = file_path
    
    return hf_models


def check_dependencies(model_name: str, hf_models: Dict[str, Path]) -> tuple:
    """Check if all dependencies for a model are available.
    
    Returns:
        tuple: (is_ready, missing_list, will_download_list)
        - is_ready: True if all deps are fully available
        - missing_list: List of truly missing dependencies
        - will_download_list: List of deps that will auto-download on first use
    """
    if model_name not in MODEL_DEPENDENCIES:
        return True, [], []
    
    deps = MODEL_DEPENDENCIES[model_name]
    missing = []
    will_download = []
    
    # Check HF repos - more lenient (check if directory exists, even if partial)
    for repo in deps.get("hf_repos", []):
        # Convert repo name to cache directory format
        cache_dir_name = f"models--{repo.replace('/', '--')}"
        repo_path = HF_CACHE_DIR / cache_dir_name
        
        if repo_path.exists():
            # Check if model.safetensors or pytorch_model.bin exists
            has_model = any(repo_path.rglob("*.safetensors")) or any(repo_path.rglob("pytorch_model.*"))
            if not has_model:
                will_download.append(f"HF repo: {repo} (partial)")
        else:
            will_download.append(f"HF repo: {repo}")
    
    # Check local files
    for local_file in deps.get("local_files", []):
        found = False
        for file_path in CHECKPOINTS_DIR.rglob(local_file):
            if file_path.is_file():
                found = True
                break
        if not found:
            missing.append(f"Local file: {local_file}")
    
    # Check vocoder checkpoint (optional - will download on first use)
    vocoder_path = deps.get("vocoder_checkpoint")
    if vocoder_path:
        full_path = Path(vocoder_path)
        if not full_path.exists():
            will_download.append(f"Vocoder: {vocoder_path}")
    
    # Model is ready only if no missing deps (will_download is OK)
    is_ready = len(missing) == 0
    
    return is_ready, missing, will_download


def detect_available_models() -> List[ModelInfo]:
    """Detect all available models and their status."""
    models = []
    
    # Find checkpoint files
    checkpoints = find_checkpoint_files()
    hf_models = find_hf_cache_models()
    
    # Check each known model
    for filename, model_meta in KNOWN_MODELS.items():
        # Check if checkpoint exists
        checkpoint_path = None
        
        # Search in checkpoints dict
        if filename in checkpoints:
            checkpoint_path = checkpoints[filename]
        else:
            # Search by partial name
            for key, path in checkpoints.items():
                if filename in key or key in filename:
                    checkpoint_path = path
                    break
        
        if checkpoint_path is None:
            # Model not found locally
            continue
        
        # Check dependencies
        deps_available, missing_deps, will_download = check_dependencies(model_meta["name"], hf_models)
        
        # Model is ready for benchmark if no missing deps and not V2
        # (will_download items are OK - they'll auto-download on first use)
        is_ready = deps_available and not model_meta.get("v2_model", False)
        
        # Create model info
        model_info = ModelInfo(
            name=model_meta["name"],
            filename=filename,
            path=str(checkpoint_path),
            size_mb=get_file_size_mb(checkpoint_path),
            description=model_meta["description"],
            estimated_params_m=model_meta["estimated_params_m"],
            config=model_meta["config"],
            vocoder=model_meta["vocoder"],
            tokenizer=model_meta["tokenizer"],
            is_v2=model_meta.get("v2_model", False),
            dependencies_available=deps_available,
            missing_dependencies=missing_deps,
            will_download=will_download,
            ready_for_benchmark=is_ready
        )
        
        models.append(model_info)
    
    return models


def print_models(models: List[ModelInfo], verbose: bool = False):
    """Print detected models in a formatted table."""
    print("\n" + "=" * 120)
    print("DETECTED MODELS")
    print("=" * 120)
    print(f"{'Model Name':<20} {'Size':>10} {'Params':>10} {'Version':>8} {'Purpose':<30} {'Status'}")
    print("-" * 120)
    
    for model in models:
        version = getattr(model, 'version', 'v1.0')
        
        if model.ready_for_benchmark:
            status_icon = "✅"
            status_text = "Ready"
        elif model.is_v2:
            status_icon = "⚠️"
            status_text = "V2 (not benchmarked)"
        elif model.missing_dependencies:
            status_icon = "❌"
            status_text = "Missing deps"
        else:
            status_icon = "🔄"
            status_text = "Will download on first use"
        
        # Extract purpose from description
        purpose = model.description.split(": ")[1][:30] if ": " in model.description else "Voice Conversion"
        
        print(f"{model.name:<20} {model.size_mb:>9.1f}MB {model.estimated_params_m:>9.0f}M "
              f"{version:>8} {purpose:<30} {status_icon} {status_text}")
        
        if verbose:
            if model.missing_dependencies:
                print(f"   ❌ Missing: {', '.join(model.missing_dependencies)}")
            if model.will_download:
                print(f"   🔄 Will download: {', '.join(model.will_download)}")
    
    print("=" * 120)
    
    # Summary
    ready = [m for m in models if m.ready_for_benchmark]
    will_download = [m for m in models if m.will_download and not m.missing_dependencies and not m.is_v2]
    missing = [m for m in models if m.missing_dependencies]
    v2 = [m for m in models if m.is_v2]
    v1 = [m for m in models if not m.is_v2]
    
    print(f"\nTotal models found: {len(models)}")
    if v1:
        print(f"  V1 models (benchmark-ready): {len(v1)}")
    if v2:
        print(f"  V2 models (not benchmarked): {len(v2)}")
    
    print(f"\n✅ Ready for benchmark: {len(ready)}")
    if will_download:
        print(f"🔄 Will download on first use: {len(will_download)}")
    if missing:
        print(f"❌ Missing dependencies: {len(missing)}")
    
    if ready:
        print(f"\n✅ Ready models: {', '.join(m.name for m in ready)}")
        print("\nYou can run benchmarks immediately:")
        print("  ./tools/examples.sh")
        print("  python tools/quick_benchmark.py")
    
    if will_download:
        print(f"\n🔄 Models that will auto-download: {', '.join(m.name for m in will_download)}")
        print("   First benchmark run will download missing dependencies automatically.")
        print("   Run real-time-gui.py once to pre-download, or just run the benchmark.")
    
    if missing:
        print(f"\n❌ Models with missing dependencies: {', '.join(m.name for m in missing)}")
        for model in missing:
            print(f"   {model.name}:")
            for dep in model.missing_dependencies:
                print(f"      - {dep}")
    
    if not ready and not will_download:
        print("\n⚠️  No models ready for benchmark.")
        print("\nTo download all models:")
        print("  1. Run: python tools/download_all_models.py --all")
        print("  2. Or download individually:")
        print("     python tools/download_all_models.py --model xlsr_tiny")
        print("     python tools/download_all_models.py --model whisper_small")
        print("     python tools/download_all_models.py --model whisper_base")
        print("\nAlternatively, run real-time-gui.py to auto-download on first use.")
        print("\nAvailable models on HuggingFace:")
        print("  - DiT_uvit_tat_xlsr_ema.pth (xlsr_tiny, 25M, V1.0)")
        print("  - DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth (whisper_small, 98M, V1.0)")
        print("  - DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth (whisper_base, 200M, V1.0)")
        print("  - DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth (v2_cfm_small, 67M, V2.0)")
        print("  - DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth (v2_ar_base, 90M, V2.0)")


def save_to_json(models: List[ModelInfo], output_path: str):
    """Save model info to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(models),
        "ready_count": sum(1 for m in models if m.ready_for_benchmark),
        "models": [asdict(m) for m in models]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Saved model info to: {output_path}")


def generate_benchmark_config(models: List[ModelInfo]) -> str:
    """Generate a Python config file for benchmark_realtime.py."""
    config_lines = [
        "# Auto-generated model configurations from detect_models.py",
        "# Do not edit manually",
        "",
        "AUTO_DETECTED_MODELS = {",
    ]
    
    for model in models:
        if model.ready_for_benchmark:
            config_lines.append(f'    "{model.name}": {{')
            config_lines.append(f'        "config": "{model.config}",')
            config_lines.append(f'        "checkpoint": "{model.filename}",')
            config_lines.append(f'        "description": "{model.description}",')
            config_lines.append(f'        "vocoder": "{model.vocoder}",')
            config_lines.append(f'        "tokenizer": "{model.tokenizer}",')
            config_lines.append(f'    }},')
    
    config_lines.append("}")
    
    return "\n".join(config_lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect available Seed-VC models")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--generate-config", action="store_true", 
                       help="Generate Python config for benchmark_realtime.py")
    
    args = parser.parse_args()
    
    print("🔍 Scanning for available models...")
    print(f"   Checkpoints directory: {CHECKPOINTS_DIR}")
    print(f"   HF cache directory: {HF_CACHE_DIR}")
    
    models = detect_available_models()
    
    if not models:
        print("\n❌ No models found!")
        print("\nTo download models:")
        print("  1. Run: python real-time-gui.py (will auto-download xlsr_tiny)")
        print("  2. Or manually download from: https://huggingface.co/Plachta/Seed-VC")
        return 1
    
    print_models(models, verbose=args.verbose)
    
    if args.output:
        save_to_json(models, args.output)
    
    if args.generate_config:
        print("\n" + "=" * 100)
        print("GENERATED CONFIG FOR benchmark_realtime.py")
        print("=" * 100)
        print(generate_benchmark_config(models))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
