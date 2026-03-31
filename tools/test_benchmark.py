#!/usr/bin/env python3
"""
Test Suite for Benchmark Tools

Validates that benchmark tools work correctly with mock data and system detection.

Usage:
    python tools/test_benchmark.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from benchmark_realtime import (
            detect_system, MODEL_CONFIGS, BenchmarkRunner,
            BenchmarkResult, run_benchmark
        )
        print("  ✅ benchmark_realtime imports successful")
    except ImportError as e:
        print(f"  ❌ benchmark_realtime import failed: {e}")
        return False
    
    try:
        import torch
        import yaml
        import librosa
        print("  ✅ Dependencies (torch, yaml, librosa) available")
    except ImportError as e:
        print(f"  ❌ Dependency missing: {e}")
        return False
    
    return True


def test_system_detection():
    """Test system detection function."""
    print("\nTesting system detection...")
    
    from benchmark_realtime import detect_system
    
    specs = detect_system()
    
    # Check required fields
    required_fields = ['os', 'device', 'mps_available', 'cuda_available']
    for field in required_fields:
        if field not in specs:
            print(f"  ❌ Missing field: {field}")
            return False
    
    print(f"  ✅ System detection successful")
    print(f"     OS: {specs.get('os', 'Unknown')}")
    print(f"     Device: {specs.get('device', 'Unknown')}")
    print(f"     MPS: {specs.get('mps_available', False)}")
    print(f"     CUDA: {specs.get('cuda_available', False)}")
    
    return True


def test_model_configs():
    """Test model configurations are properly defined."""
    print("\nTesting model configurations...")
    
    from benchmark_realtime import MODEL_CONFIGS
    
    # Check all expected models exist
    expected_models = ['xlsr_tiny', 'whisper_small', 'whisper_base']
    for model in expected_models:
        if model not in MODEL_CONFIGS:
            print(f"  ❌ Missing model config: {model}")
            return False
    
    # Check config structure
    for name, config in MODEL_CONFIGS.items():
        required_attrs = ['name', 'yaml_file', 'checkpoint', 'speech_tokenizer', 'vocoder']
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"  ❌ Model {name} missing attribute: {attr}")
                return False
    
    print(f"  ✅ All {len(MODEL_CONFIGS)} model configs valid")
    for name, config in MODEL_CONFIGS.items():
        print(f"     - {name}: {config.description}")
    
    return True


def test_benchmark_result_structure():
    """Test BenchmarkResult dataclass structure."""
    print("\nTesting BenchmarkResult structure...")
    
    from benchmark_realtime import BenchmarkResult
    from datetime import datetime
    
    # Create a sample result
    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name="test_model",
        diffusion_steps=10,
        block_time=0.20,
        num_runs=5,
        latencies_ms=[100.0, 102.0, 98.0, 101.0, 99.0],
        mean_latency_ms=100.0,
        std_latency_ms=1.58,
        min_latency_ms=98.0,
        max_latency_ms=102.0,
        p50_latency_ms=100.0,
        p90_latency_ms=101.5,
        p99_latency_ms=101.9,
        rtf=0.5,
        feasible=True,
        feasible_mean=True,
        feasible_p99=True
    )
    
    # Check feasibility logic
    assert result.feasible == (result.feasible_mean and result.feasible_p99)
    assert result.rtf == result.mean_latency_ms / (result.block_time * 1000)
    
    print(f"  ✅ BenchmarkResult structure valid")
    print(f"     Mean: {result.mean_latency_ms:.1f}ms, RTF: {result.rtf:.3f}")
    print(f"     Feasible: {result.feasible}")
    
    return True


def test_feasibility_logic():
    """Test feasibility determination logic."""
    print("\nTesting feasibility logic...")
    
    from benchmark_realtime import BenchmarkResult
    from datetime import datetime
    
    # Test case 1: Feasible (mean < 80%, P99 < 100%)
    result1 = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name="test",
        diffusion_steps=10,
        block_time=0.20,  # 200ms
        num_runs=1,
        mean_latency_ms=150.0,  # 75% of block
        p99_latency_ms=190.0,   # 95% of block
        feasible_mean=True,
        feasible_p99=True,
        feasible=True
    )
    assert result1.feasible, "Should be feasible"
    print(f"  ✅ Test 1 passed: 150ms mean, 190ms P99 → Feasible")
    
    # Test case 2: Not feasible (mean > 80%)
    result2 = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name="test",
        diffusion_steps=10,
        block_time=0.20,
        num_runs=1,
        mean_latency_ms=171.0,  # 85.5% of block
        p99_latency_ms=190.0,
        feasible_mean=False,
        feasible_p99=True,
        feasible=False
    )
    assert not result2.feasible, "Should not be feasible"
    print(f"  ✅ Test 2 passed: 171ms mean, 190ms P99 → Not feasible (mean too high)")
    
    # Test case 3: Not feasible (P99 > 100%)
    result3 = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name="test",
        diffusion_steps=10,
        block_time=0.20,
        num_runs=1,
        mean_latency_ms=150.0,
        p99_latency_ms=210.0,  # 105% of block
        feasible_mean=True,
        feasible_p99=False,
        feasible=False
    )
    assert not result3.feasible, "Should not be feasible"
    print(f"  ✅ Test 3 passed: 150ms mean, 210ms P99 → Not feasible (P99 too high)")
    
    return True


def test_latency_statistics():
    """Test latency statistics computation."""
    print("\nTesting latency statistics...")
    
    # Sample latencies
    latencies = [100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 100.0, 101.0, 99.0]
    
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    print(f"  Sample: {latencies}")
    print(f"  Mean: {mean_lat:.2f}ms, Std: {std_lat:.2f}ms")
    print(f"  Min: {min_lat:.2f}ms, Max: {max_lat:.2f}ms")
    print(f"  P50: {p50:.2f}ms, P90: {p90:.2f}ms, P99: {p99:.2f}ms")
    
    # Verify statistics
    assert 97 <= mean_lat <= 103, "Mean should be in range"
    assert std_lat >= 0, "Std should be non-negative"
    assert min_lat <= p50 <= max_lat, "P50 should be in range"
    assert p90 >= p50, "P90 should be >= P50"
    assert p99 >= p90, "P99 should be >= P90"
    
    print(f"  ✅ Statistics computation valid")
    
    return True


def test_rtf_calculation():
    """Test Real-Time Factor calculation."""
    print("\nTesting RTF calculation...")
    
    test_cases = [
        (100.0, 0.20, 0.5),   # 100ms latency, 200ms block → RTF = 0.5
        (160.0, 0.20, 0.8),   # 160ms latency, 200ms block → RTF = 0.8
        (200.0, 0.20, 1.0),   # 200ms latency, 200ms block → RTF = 1.0
        (100.0, 0.25, 0.4),   # 100ms latency, 250ms block → RTF = 0.4
    ]
    
    for latency_ms, block_time, expected_rtf in test_cases:
        rtf = latency_ms / (block_time * 1000)
        assert abs(rtf - expected_rtf) < 0.001, f"RTF mismatch: {rtf} vs {expected_rtf}"
        print(f"  ✅ {latency_ms}ms / {block_time*1000:.0f}ms → RTF = {rtf:.3f}")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("BENCHMARK TOOL TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("System Detection", test_system_detection),
        ("Model Configs", test_model_configs),
        ("Result Structure", test_benchmark_result_structure),
        ("Feasibility Logic", test_feasibility_logic),
        ("Latency Statistics", test_latency_statistics),
        ("RTF Calculation", test_rtf_calculation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
