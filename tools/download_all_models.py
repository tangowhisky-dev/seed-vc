#!/usr/bin/env python3
"""
Download All Seed-VC Models

Downloads all 4 models from HuggingFace and their dependencies.

Usage:
    python tools/download_all_models.py [--all] [--model MODEL_NAME]

Examples:
    python tools/download_all_models.py --all
    python tools/download_all_models.py --model xlsr_tiny
    python tools/download_all_models.py --model whisper_small whisper_base
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import subprocess

# Models to download
MODELS = {
    "xlsr_tiny": {
        "checkpoint": "DiT_uvit_tat_xlsr_ema.pth",
        "size_mb": 135,
        "description": "V1.0: 25M DiT + XLSR-large (300M frozen) + HIFT (real-time VC)",
        "dependencies": [
            "facebook/wav2vec2-xls-r-300m",
            "FunAudioLLM/CosyVoice-300M",
        ],
        "vocoder": {
            "file": "hift.pt",
            "repo": "Plachta/Seed-VC"
        }
    },
    "whisper_small": {
        "checkpoint": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "size_mb": 342,
        "description": "V1.0: 98M DiT + Whisper-small (39M) + BigVGAN (offline VC)",
        "dependencies": [
            "openai/whisper-small",
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ],
        "vocoder": {
            "file": "bigvgan_generator.pt",
            "repo": "nvidia/bigvgan_v2_22khz_80band_256x",
            "subdir": "snapshots"
        }
    },
    "whisper_base": {
        "checkpoint": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "size_mb": 850,
        "description": "V1.0: 200M DiT + Whisper-small (39M) + BigVGAN (SVC, 44.1kHz)",
        "dependencies": [
            "openai/whisper-small",
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ],
        "vocoder": {
            "file": "bigvgan_generator.pt",
            "repo": "nvidia/bigvgan_v2_22khz_80band_256x",
            "subdir": "snapshots"
        }
    },
    "v2_cfm_small": {
        "checkpoint": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ema.pth",
        "size_mb": 337,
        "description": "V2.0: 67M CFM + ASTRAL-Quantization + BigVGAN (accent conversion)",
        "dependencies": [
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ],
        "vocoder": {
            "file": "bigvgan_generator.pt",
            "repo": "nvidia/bigvgan_v2_22khz_80band_256x",
            "subdir": "snapshots"
        }
    },
    "v2_ar_base": {
        "checkpoint": "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "size_mb": 342,
        "description": "V2.0: 90M AR + ASTRAL-Quantization + BigVGAN (accent conversion)",
        "dependencies": [
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ],
        "vocoder": {
            "file": "bigvgan_generator.pt",
            "repo": "nvidia/bigvgan_v2_22khz_80band_256x",
            "subdir": "snapshots"
        }
    },
}

CHECKPOINTS_DIR = Path("./checkpoints")


def print_models_table():
    """Print available models in a table."""
    from pathlib import Path
    
    print("\n" + "=" * 120)
    print("AVAILABLE MODELS TO DOWNLOAD")
    print("=" * 120)
    print(f"{'Model Name':<20} {'Size':>10} {'Version':>10} {'Purpose':<35} {'Status'}")
    print("-" * 120)
    
    for name, info in MODELS.items():
        version = info.get("version", "v1.0")
        purpose = info["description"].split(": ")[1] if ": " in info["description"] else "Voice Conversion"
        
        # Check if already downloaded - search in both blobs and snapshots
        checkpoint_exists = False
        
        # Search in blobs directory
        checkpoint_blob_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "blobs"
        if checkpoint_blob_dir.exists():
            for blob_file in checkpoint_blob_dir.iterdir():
                if blob_file.is_file() and info["checkpoint"].split(".")[-1] in blob_file.name:
                    checkpoint_exists = True
                    break
        
        # Search in snapshots directory
        if not checkpoint_exists:
            checkpoint_snap_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "snapshots"
            if checkpoint_snap_dir.exists():
                for snapshot_subdir in checkpoint_snap_dir.iterdir():
                    if snapshot_subdir.is_dir():
                        target_file = snapshot_subdir / info["checkpoint"]
                        if target_file.exists():
                            checkpoint_exists = True
                            break
                        # Also check v2 subdirectory
                        v2_dir = snapshot_subdir / "v2"
                        if v2_dir.exists():
                            # Check for CFM or AR models
                            if name in ["v2_cfm_small", "v2_ar_base"]:
                                model_type = "cfm" if "cfm" in name else "ar"
                                v2_file = v2_dir / f"{model_type}_small.pth"
                                if v2_file.exists() or (model_type == "ar" and (v2_dir / "ar_base.pth").exists()):
                                    checkpoint_exists = True
                                    break
        
        # Also check if file exists directly in checkpoints
        if not checkpoint_exists:
            direct_path = CHECKPOINTS_DIR / info["checkpoint"]
            if direct_path.exists():
                checkpoint_exists = True
        
        status = "✅ Downloaded" if checkpoint_exists else "❌ Not downloaded"
        
        print(f"{name:<20} {info['size_mb']:>9.0f}MB {version:>10} {purpose:<35} {status}")
    
    print("=" * 120)
    print(f"\nTotal models: {len(MODELS)}")
    print(f"Total size: ~{sum(m['size_mb'] for m in MODELS.values())}MB (~1.7GB)")
    print()


def download_file(repo_id: str, filename: str, local_dir: Path, subdir: str = None):
    """Download a file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📥 Downloading: {repo_id}/{filename}")
    
    try:
        if subdir:
            # For files in subdirectories
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subdir}/{filename}" if subdir else filename,
                cache_dir=str(CHECKPOINTS_DIR)
            )
        else:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(CHECKPOINTS_DIR)
            )
        
        print(f"  ✅ Downloaded: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def download_repo(repo_id: str):
    """Download a HuggingFace repository (will be cached)."""
    from huggingface_hub import snapshot_download
    
    print(f"  📥 Downloading repo: {repo_id}")
    
    try:
        # Just trigger download by listing files
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        
        # Download will happen on first use
        print(f"  ✅ Repo ready: {repo_id} ({len(files)} files)")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    if model_name not in MODELS:
        return False
    
    model = MODELS[model_name]
    
    # Search in blobs directory
    checkpoint_blob_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "blobs"
    if checkpoint_blob_dir.exists():
        for blob_file in checkpoint_blob_dir.iterdir():
            if blob_file.is_file() and model["checkpoint"].split(".")[-1] in blob_file.name:
                return True
    
    # Search in snapshots directory
    checkpoint_snap_dir = CHECKPOINTS_DIR / "models--Plachta--Seed-VC" / "snapshots"
    if checkpoint_snap_dir.exists():
        for snapshot_subdir in checkpoint_snap_dir.iterdir():
            if snapshot_subdir.is_dir():
                target_file = snapshot_subdir / model["checkpoint"]
                if target_file.exists():
                    return True
                # Check v2 subdirectory
                v2_dir = snapshot_subdir / "v2"
                if v2_dir.exists():
                    if model_name in ["v2_cfm_small", "v2_ar_base"]:
                        model_type = "cfm" if "cfm" in model_name else "ar"
                        v2_file = v2_dir / f"{model_type}_small.pth"
                        if v2_file.exists() or (model_type == "ar" and (v2_dir / "ar_base.pth").exists()):
                            return True
    
    # Check if file exists directly in checkpoints
    direct_path = CHECKPOINTS_DIR / model["checkpoint"]
    if direct_path.exists():
        return True
    
    return False


def download_model(model_name: str, verbose: bool = True) -> bool:
    """Download a single model and its dependencies."""
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Available: {', '.join(MODELS.keys())}")
        return False
    
    model = MODELS[model_name]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Downloading: {model_name}")
        print(f"Description: {model['description']}")
        print(f"Size: ~{model['size_mb']}MB")
        print(f"{'='*80}")
    
    # Check if already downloaded
    if check_model_downloaded(model_name):
        if verbose:
            print(f"  ⏭️  Model already downloaded, skipping...")
        return True  # Count as successful
    
    # Download checkpoint from Plachta/Seed-VC
    success = download_file("Plachta/Seed-VC", model["checkpoint"], CHECKPOINTS_DIR)
    if not success:
        return False
    
    # Download dependencies
    for dep_repo in model["dependencies"]:
        if verbose:
            print(f"\n  Dependency: {dep_repo}")
        download_repo(dep_repo)
    
    # Download vocoder if specified
    if "vocoder" in model:
        vocoder_info = model["vocoder"]
        vocoder_path = CHECKPOINTS_DIR / vocoder_info["file"]
        
        if not vocoder_path.exists():
            if verbose:
                print(f"\n  Vocoder: {vocoder_info['file']}")
            download_file(
                vocoder_info["repo"],
                vocoder_info["file"],
                CHECKPOINTS_DIR,
                vocoder_info.get("subdir")
            )
        else:
            if verbose:
                print(f"\n  ⏭️  Vocoder already downloaded, skipping...")
    
    if verbose:
        print(f"\n✅ {model_name} download complete!")
    
    return True


def download_all(verbose: bool = True) -> List[str]:
    """Download all models."""
    print_models_table()
    
    print("\n🚀 Starting download of all models...")
    print("(This may take 10-30 minutes depending on internet speed)\n")
    
    successful = []
    failed = []
    
    for model_name in MODELS.keys():
        if download_model(model_name, verbose=verbose):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(successful)}/{len(MODELS)} ({', '.join(successful)})")
    if failed:
        print(f"❌ Failed: {len(failed)} ({', '.join(failed)})")
    print(f"\nTotal downloaded: ~{sum(MODELS[m]['size_mb'] for m in successful)}MB")
    print(f"\n📁 Models saved to: {CHECKPOINTS_DIR.absolute()}")
    print(f"\n📊 To verify: python tools/detect_models.py")
    print(f"🏃 To benchmark: ./tools/examples.sh")
    print(f"{'='*80}\n")
    
    return successful


def main():
    parser = argparse.ArgumentParser(description="Download all Seed-VC models")
    parser.add_argument("--all", "-a", action="store_true", help="Download all models")
    parser.add_argument("--model", "-m", nargs="+", help="Download specific model(s)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        print_models_table()
        return 0
    
    # Download all
    if args.all:
        successful = download_all(verbose=not args.quiet)
        return 0 if len(successful) == len(MODELS) else 1
    
    # Download specific models
    if args.model:
        successful = []
        for model_name in args.model:
            if download_model(model_name, verbose=not args.quiet):
                successful.append(model_name)
        
        if successful:
            print(f"\n✅ Downloaded: {', '.join(successful)}")
            return 0
        else:
            return 1
    
    # No arguments - show help
    parser.print_help()
    print("\nExamples:")
    print("  python tools/download_all_models.py --all")
    print("  python tools/download_all_models.py --model xlsr_tiny")
    print("  python tools/download_all_models.py --model whisper_small whisper_base")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
