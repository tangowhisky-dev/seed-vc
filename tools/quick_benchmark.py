#!/usr/bin/env python3
"""
Quick Real-Time Feasibility Checker

A simplified benchmark that quickly validates if a model can run in real-time
on the current system. Runs fewer iterations and provides a fast feasibility check.

Usage:
    python tools/quick_benchmark.py [--model MODEL_NAME] [--steps STEPS]
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import librosa
import torch

# Import from main benchmark
from benchmark_realtime import (
    detect_system, MODEL_CONFIGS, run_benchmark, 
    print_system_specs, print_result, print_summary
)


def main():
    parser = argparse.ArgumentParser(
        description="Quick Real-Time Feasibility Checker for Seed-VC"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_CONFIGS.keys()),
        default="xlsr_tiny",
        help="Model to benchmark (default: xlsr_tiny)"
    )
    
    parser.add_argument(
        "--steps", "-s",
        type=str,
        default="6",
        help="Diffusion steps to test (default: 6)"
    )
    
    parser.add_argument(
        "--block-time", "-b",
        type=float,
        default=0.20,
        help="Block time in seconds (default: 0.20)"
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of runs (default: 3 for quick check)"
    )
    
    parser.add_argument(
        "--test-audio",
        type=str,
        default="examples/source/source_1.wav",
        help="Path to test audio file"
    )
    
    parser.add_argument(
        "--reference-audio",
        type=str,
        default="examples/reference/ref_1.wav",
        help="Path to reference audio file"
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    # Detect system
    print("🔍 Detecting system...")
    system_specs = detect_system()
    print_system_specs(system_specs)
    
    # Check device
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("❌ Error: No GPU available (MPS or CUDA required)")
        sys.exit(1)
    
    # Load audio
    print(f"📁 Loading audio files...")
    if not Path(args.test_audio).exists() or not Path(args.reference_audio).exists():
        print(f"❌ Audio files not found")
        sys.exit(1)
    
    test_wav, test_sr = librosa.load(args.test_audio, sr=None)
    ref_wav, ref_sr = librosa.load(args.reference_audio, sr=None)
    print(f"   Test: {len(test_wav)/test_sr:.1f}s, Reference: {len(ref_wav)/ref_sr:.1f}s")
    
    # Run benchmark
    model_config = MODEL_CONFIGS[args.model]
    print(f"\n🚀 Benchmarking {model_config.name}...")
    print(f"   Description: {model_config.description}")
    print(f"   Steps: {steps}, Block time: {args.block_time}s")
    
    results = []
    for steps in steps:
        result = run_benchmark(
            model_config=model_config,
            diffusion_steps=steps,
            block_time=args.block_time,
            test_audio=(test_wav, test_sr),
            reference_audio=(ref_wav, ref_sr),
            system_specs=system_specs,
            num_runs=args.runs
        )
        results.append(result)
        print_result(result)
    
    # Summary
    print_summary(results)
    
    # Verdict
    feasible = [r for r in results if r.feasible]
    print("\n" + "=" * 60)
    if feasible:
        best = min(feasible, key=lambda x: x.rtf)
        print(f"✅ VERDICT: {model_config.name} is FEASIBLE for real-time")
        print(f"   Recommended: {best.diffusion_steps} steps "
              f"(Latency: {best.mean_latency_ms:.0f}ms, RTF: {best.rtf:.3f})")
    else:
        print(f"❌ VERDICT: {model_config.name} is NOT FEASIBLE for real-time")
        print(f"   Try: reducing diffusion steps or increasing block time")
    print("=" * 60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"quick_benchmark_{args.model}_{timestamp}.json"
    import json
    from dataclasses import asdict
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
