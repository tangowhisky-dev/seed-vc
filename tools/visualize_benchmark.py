#!/usr/bin/env python3
"""
Benchmark Results Visualizer

Generate plots and analysis from benchmark JSON results.
Requires: matplotlib, seaborn

Usage:
    python tools/visualize_benchmark.py results.json [--output OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("❌ Required packages not installed:")
    print("   pip install matplotlib seaborn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(json_path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_feasibility_heatmap(results: list, output_path: Path):
    """Create heatmap showing feasibility across models and steps."""
    
    # Extract unique models and steps
    models = sorted(set(r['model_name'] for r in results))
    steps = sorted(set(r['diffusion_steps'] for r in results))
    
    # Create feasibility matrix
    data = np.zeros((len(models), len(steps)))
    for i, model in enumerate(models):
        for j, step in enumerate(steps):
            matching = [r for r in results if r['model_name'] == model and r['diffusion_steps'] == step]
            if matching:
                data[i, j] = 1 if matching[0]['feasible'] else 0
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data, 
        annot=True, 
        fmt='.0f',
        xticklabels=steps,
        yticklabels=[m.replace('_', '-') for m in models],
        cmap='RdYlGn',
        vmin=0, vmax=1,
        cbar_kws={'label': 'Feasible (1=Yes, 0=No)'}
    )
    plt.title('Real-Time Feasibility by Model and Diffusion Steps')
    plt.xlabel('Diffusion Steps')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_latency_bar_chart(results: list, output_path: Path):
    """Create bar chart showing latency by model and steps."""
    
    # Group by model and steps
    grouped = {}
    for r in results:
        if r['error']:
            continue
        key = (r['model_name'], r['diffusion_steps'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].extend(r['latencies_ms'])
    
    # Compute statistics
    means = []
    stds = []
    labels = []
    colors = []
    
    for (model, steps), latencies in sorted(grouped.items()):
        means.append(np.mean(latencies))
        stds.append(np.std(latencies))
        labels.append(f"{model}\n{steps} steps")
        colors.append('green' if any(r['feasible'] for r in results 
                                     if r['model_name'] == model and r['diffusion_steps'] == steps) 
                     else 'red')
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(means)), means, yerr=stds, color=colors, capsize=5)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Mean Latency by Model Configuration (±1 std)')
    plt.axhline(y=160, color='orange', linestyle='--', label='80% of 200ms block')
    plt.axhline(y=200, color='red', linestyle='--', label='100% of 200ms block')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_component_breakdown(results: list, output_path: Path):
    """Create stacked bar chart showing component latencies."""
    
    # Filter to one config per model (use first)
    seen_models = set()
    filtered = []
    for r in results:
        if r['model_name'] not in seen_models and r['component_latencies']:
            filtered.append(r)
            seen_models.add(r['model_name'])
    
    if not filtered:
        print("⚠️  No component latency data found")
        return
    
    # Components to show
    components = ['content_encoder', 'dit', 'vocoder', 'length_regulator', 'vad']
    component_labels = {
        'content_encoder': 'Content Encoder',
        'dit': 'DiT Inference',
        'vocoder': 'Vocoder',
        'length_regulator': 'Length Regulator',
        'vad': 'VAD'
    }
    
    # Build data
    models = []
    component_data = {c: [] for c in components}
    
    for r in filtered:
        models.append(r['model_name'].replace('_', '-'))
        comp = r['component_latencies']
        for c in components:
            component_data[c].append(comp.get(c, 0))
    
    # Plot stacked bar
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(models))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, (comp, color) in enumerate(zip(components, colors)):
        plt.bar(
            range(len(models)), 
            component_data[comp], 
            bottom=bottom, 
            color=color,
            label=component_labels[comp]
        )
        bottom += np.array(component_data[comp])
    
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Component Latency Breakdown')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_rtf_comparison(results: list, output_path: Path):
    """Create RTF comparison chart."""
    
    # Filter feasible results
    feasible = [r for r in results if r['feasible'] and not r['error']]
    
    if not feasible:
        print("⚠️  No feasible configurations to compare")
        return
    
    # Group by model
    model_rtfs = {}
    for r in feasible:
        if r['model_name'] not in model_rtfs:
            model_rtfs[r['model_name']] = []
        model_rtfs[r['model_name']].append((r['diffusion_steps'], r['rtf']))
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_rtfs)))
    
    for i, (model, data) in enumerate(sorted(model_rtfs.items())):
        steps = [d[0] for d in data]
        rtfs = [d[1] for d in data]
        plt.plot(steps, rtfs, 'o-', label=model.replace('_', '-'), color=colors[i], linewidth=2, markersize=8)
    
    plt.axhline(y=0.8, color='orange', linestyle='--', label='RTF = 0.8 (threshold)')
    plt.axhline(y=1.0, color='red', linestyle='--', label='RTF = 1.0 (real-time)')
    plt.xlabel('Diffusion Steps')
    plt.ylabel('Real-Time Factor (RTF)')
    plt.title('RTF Comparison: Lower is Better')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")


def create_latency_distribution(results: list, output_path: Path):
    """Create latency distribution histogram."""
    
    # Collect all latencies
    all_latencies = []
    for r in results:
        if not r['error'] and r['latencies_ms']:
            all_latencies.extend(r['latencies_ms'])
    
    if not all_latencies:
        print("⚠️  No latency data found")
        return
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_latencies, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(all_latencies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_latencies):.1f}ms')
    plt.axvline(x=np.percentile(all_latencies, 99), color='orange', linestyle='--',
                label=f'P99: {np.percentile(all_latencies, 99):.1f}ms')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution Across All Benchmarks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_path}")


def generate_summary_report(results: list, output_path: Path):
    """Generate text summary report."""
    
    report = []
    report.append("=" * 80)
    report.append("BENCHMARK RESULTS SUMMARY")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total configurations tested: {len(results)}")
    
    # System info
    if results:
        specs = results[0].get('system_specs', {})
        report.append(f"\nSystem: {specs.get('chip', 'Unknown')}")
        report.append(f"Device: {specs.get('device', 'Unknown')}")
        report.append(f"Memory: {specs.get('memory_total_gb', 0):.1f}GB")
    
    # Feasibility summary
    feasible = [r for r in results if r.get('feasible', False)]
    report.append(f"\nFeasible configurations: {len(feasible)}/{len(results)}")
    
    if feasible:
        report.append("\n✅ FEASIBLE CONFIGURATIONS:")
        report.append("-" * 60)
        for r in sorted(feasible, key=lambda x: x.get('rtf', float('inf'))):
            report.append(f"  {r['model_name']:20s} {r['diffusion_steps']:3d} steps: "
                         f"Mean={r['mean_latency_ms']:6.0f}ms, RTF={r['rtf']:.3f}")
        
        # Best config
        best = min(feasible, key=lambda x: x.get('rtf', float('inf')))
        report.append("\n🏆 BEST CONFIGURATION:")
        report.append(f"  Model: {best['model_name']}")
        report.append(f"  Steps: {best['diffusion_steps']}")
        report.append(f"  Latency: {best['mean_latency_ms']:.0f}ms (P99: {best['p99_latency_ms']:.0f}ms)")
        report.append(f"  RTF: {best['rtf']:.3f}")
    
    # Component analysis
    report.append("\n📊 COMPONENT ANALYSIS (avg across feasible configs):")
    report.append("-" * 60)
    if feasible:
        comp_totals = {}
        comp_counts = {}
        for r in feasible:
            comp = r.get('component_latencies', {})
            for key, value in comp.items():
                if isinstance(value, (int, float)):
                    comp_totals[key] = comp_totals.get(key, 0) + value
                    comp_counts[key] = comp_counts.get(key, 0) + 1
        
        for comp in sorted(comp_totals.keys()):
            avg = comp_totals[comp] / comp_counts[comp]
            report.append(f"  {comp:25s}: {avg:.1f}ms")
    
    # Errors
    errors = [r for r in results if r.get('error')]
    if errors:
        report.append(f"\n❌ ERRORS ({len(errors)} configurations):")
        report.append("-" * 60)
        for r in errors:
            report.append(f"  {r['model_name']} ({r['diffusion_steps']} steps): {r['error'][:50]}...")
    
    report.append("\n" + "=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Saved: {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_json", help="Path to benchmark results JSON file")
    parser.add_argument("--output", "-o", default="benchmark_visualization",
                       help="Output directory for plots (default: benchmark_visualization)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation, only create text report")
    
    args = parser.parse_args()
    
    # Load results
    if not Path(args.results_json).exists():
        print(f"❌ Results file not found: {args.results_json}")
        sys.exit(1)
    
    print(f"📊 Loading results from: {args.results_json}")
    results = load_results(args.results_json)
    print(f"   Found {len(results)} benchmark results")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Generate text report
    print("\n📝 Generating summary report...")
    report = generate_summary_report(results, output_dir / "SUMMARY.md")
    print(report)
    
    # Generate plots
    if not args.no_plots:
        print("\n📈 Generating plots...")
        try:
            create_feasibility_heatmap(results, output_dir / "feasibility_heatmap.png")
            create_latency_bar_chart(results, output_dir / "latency_comparison.png")
            create_component_breakdown(results, output_dir / "component_breakdown.png")
            create_rtf_comparison(results, output_dir / "rtf_comparison.png")
            create_latency_distribution(results, output_dir / "latency_distribution.png")
            print(f"\n✅ All plots saved to: {output_dir}/")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
            print("   Install matplotlib and seaborn for visualization:")
            print("   pip install matplotlib seaborn")
    
    print(f"\n🎉 Visualization complete!")
    print(f"   Summary: {output_dir}/SUMMARY.md")
    if not args.no_plots:
        print(f"   Plots: {output_dir}/*.png")


if __name__ == "__main__":
    main()
