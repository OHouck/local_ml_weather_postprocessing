#!/usr/bin/env python3
"""
Analyze Architecture Experiment Results

Parses log files and creates comparison report for the 4 architectures:
- MLP_Deep
- MLP_Wide
- UNet_Light
- UNet_Deep
"""

import os
import re
import json
import glob
from pathlib import Path
import numpy as np

# Experiment configuration
EXPERIMENTS = {
    'mlp_deep': {
        'name': 'MLP Deep',
        'description': 'Deep MLP (6 layers × 1024 neurons)',
        'architecture': 'MLP'
    },
    'mlp_wide': {
        'name': 'MLP Wide',
        'description': 'Wide MLP (3 layers × 2048 neurons)',
        'architecture': 'MLP'
    },
    'unet_light': {
        'name': 'UNet Light',
        'description': 'Lightweight UNet (64 base channels)',
        'architecture': 'UNet'
    },
    'unet_deep': {
        'name': 'UNet Deep',
        'description': 'Deep UNet (128 base channels)',
        'architecture': 'UNet'
    }
}

LEAD_TIMES = [24, 72, 144]  # hours


def parse_log_file(log_path):
    """Parse a log file to extract MSE results and training time"""

    results = {
        'lead_times': {},
        'training_time': None,
        'architecture_params': {}
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract MSE for each lead time
        mse_pattern = r'Lead time (\d+)h - MSE original: ([\d.]+), MSE corrected: ([\d.]+)'
        for match in re.finditer(mse_pattern, content):
            lead_time = int(match.group(1))
            mse_original = float(match.group(2))
            mse_corrected = float(match.group(3))

            rmse_original = np.sqrt(mse_original)
            rmse_corrected = np.sqrt(mse_corrected)
            improvement = (rmse_original - rmse_corrected) / rmse_original * 100

            results['lead_times'][lead_time] = {
                'mse_original': mse_original,
                'mse_corrected': mse_corrected,
                'rmse_original': rmse_original,
                'rmse_corrected': rmse_corrected,
                'improvement_percent': improvement
            }

        # Extract training time
        time_pattern = r'Training complete in ([\d.]+) minutes'
        match = re.search(time_pattern, content)
        if match:
            results['training_time'] = float(match.group(1))

        # Extract architecture parameters
        # MLP
        mlp_hidden = re.search(r'MLP hidden_dim: (\d+)', content)
        mlp_layers = re.search(r'MLP num_layers: (\d+)', content)
        mlp_dropout = re.search(r'MLP dropout: ([\d.]+)', content)

        if mlp_hidden:
            results['architecture_params']['hidden_dim'] = int(mlp_hidden.group(1))
            results['architecture_params']['num_layers'] = int(mlp_layers.group(1))
            results['architecture_params']['dropout'] = float(mlp_dropout.group(1))

        # UNet
        unet_hidden = re.search(r'UNet hidden_dim: (\d+)', content)
        unet_dropout = re.search(r'UNet dropout: ([\d.]+)', content)

        if unet_hidden:
            results['architecture_params']['hidden_dim'] = int(unet_hidden.group(1))
            results['architecture_params']['dropout'] = float(unet_dropout.group(1))

        return results

    except FileNotFoundError:
        print(f"  Warning: Log file not found: {log_path}")
        return None
    except Exception as e:
        print(f"  Error parsing {log_path}: {e}")
        return None


def find_latest_log(log_dir, exp_prefix):
    """Find the most recent log file for an experiment"""
    pattern = os.path.join(log_dir, f"{exp_prefix}_*.log")
    log_files = glob.glob(pattern)

    if not log_files:
        return None

    # Return the most recent file
    return max(log_files, key=os.path.getmtime)


def create_comparison_report(results_dict):
    """Create a comprehensive comparison report"""

    report = []
    report.append("=" * 80)
    report.append("ARCHITECTURE EXPERIMENT RESULTS")
    report.append("=" * 80)
    report.append("")
    report.append("Objective: Improve Pangu weather forecasts for India region")
    report.append("Region: India (6x6 degree subregion)")
    report.append("Output variable: 2m temperature")
    report.append("Lead times: 24h, 72h, 144h")
    report.append("")

    # Individual experiment results
    report.append("=" * 80)
    report.append("INDIVIDUAL EXPERIMENT RESULTS")
    report.append("=" * 80)
    report.append("")

    for exp_name, exp_info in EXPERIMENTS.items():
        results = results_dict.get(exp_name)

        report.append(f"{exp_info['name']} - {exp_info['description']}")
        report.append("-" * 80)

        if results is None:
            report.append("  ⚠ No results found")
            report.append("")
            continue

        # Architecture parameters
        if results['architecture_params']:
            report.append("  Architecture Parameters:")
            for param, value in results['architecture_params'].items():
                report.append(f"    {param}: {value}")

        # Training time
        if results['training_time']:
            report.append(f"  Training Time: {results['training_time']:.2f} minutes")

        # Results by lead time
        report.append("")
        report.append("  Results by Lead Time:")
        report.append(f"    {'Lead Time':<12} {'RMSE Orig':<12} {'RMSE Corr':<12} {'Improvement':<12}")
        report.append(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for lead_time in LEAD_TIMES:
            if lead_time in results['lead_times']:
                lt_results = results['lead_times'][lead_time]
                report.append(
                    f"    {lead_time}h{'':<9} "
                    f"{lt_results['rmse_original']:<12.6f} "
                    f"{lt_results['rmse_corrected']:<12.6f} "
                    f"{lt_results['improvement_percent']:>10.2f}%"
                )

        report.append("")
        report.append("")

    # Comparison table
    report.append("=" * 80)
    report.append("COMPARISON SUMMARY")
    report.append("=" * 80)
    report.append("")

    # By lead time
    for lead_time in LEAD_TIMES:
        report.append(f"Lead Time: {lead_time}h")
        report.append("-" * 80)
        report.append(f"{'Architecture':<20} {'RMSE Original':<15} {'RMSE Corrected':<15} {'Improvement':<15}")
        report.append(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        lt_results = []
        for exp_name, exp_info in EXPERIMENTS.items():
            results = results_dict.get(exp_name)
            if results and lead_time in results['lead_times']:
                lt_data = results['lead_times'][lead_time]
                lt_results.append({
                    'name': exp_info['name'],
                    'rmse_original': lt_data['rmse_original'],
                    'rmse_corrected': lt_data['rmse_corrected'],
                    'improvement': lt_data['improvement_percent']
                })

        # Sort by improvement (best first)
        lt_results.sort(key=lambda x: x['improvement'], reverse=True)

        for i, res in enumerate(lt_results):
            marker = "⭐" if i == 0 else "  "
            report.append(
                f"{marker} {res['name']:<18} "
                f"{res['rmse_original']:<15.6f} "
                f"{res['rmse_corrected']:<15.6f} "
                f"{res['improvement']:>13.2f}%"
            )

        report.append("")

    # Overall best
    report.append("=" * 80)
    report.append("BEST ARCHITECTURES")
    report.append("=" * 80)
    report.append("")

    # Calculate average improvement across all lead times
    avg_improvements = {}
    for exp_name, exp_info in EXPERIMENTS.items():
        results = results_dict.get(exp_name)
        if results:
            improvements = [
                results['lead_times'][lt]['improvement_percent']
                for lt in LEAD_TIMES if lt in results['lead_times']
            ]
            if improvements:
                avg_improvements[exp_name] = {
                    'name': exp_info['name'],
                    'description': exp_info['description'],
                    'avg_improvement': np.mean(improvements),
                    'improvements': improvements
                }

    # Sort by average improvement
    sorted_exps = sorted(avg_improvements.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)

    report.append(f"{'Rank':<6} {'Architecture':<20} {'Avg Improvement':<20} {'Description'}")
    report.append(f"{'-'*6} {'-'*20} {'-'*20} {'-'*40}")

    for rank, (exp_name, data) in enumerate(sorted_exps, 1):
        marker = "🏆" if rank == 1 else f"{rank}."
        report.append(
            f"{marker:<6} {data['name']:<20} "
            f"{data['avg_improvement']:>18.2f}%  {data['description']}"
        )

    if sorted_exps:
        report.append("")
        report.append("Detailed improvements by lead time:")
        for rank, (exp_name, data) in enumerate(sorted_exps, 1):
            report.append(f"  {rank}. {data['name']}:")
            for i, lt in enumerate(LEAD_TIMES):
                if i < len(data['improvements']):
                    report.append(f"      {lt}h: {data['improvements'][i]:>6.2f}%")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def create_json_summary(results_dict):
    """Create a JSON summary of results"""
    summary = {
        'timestamp': str(Path.ctime(Path.home())),
        'experiments': {}
    }

    for exp_name, exp_info in EXPERIMENTS.items():
        results = results_dict.get(exp_name)
        if results:
            summary['experiments'][exp_name] = {
                'name': exp_info['name'],
                'description': exp_info['description'],
                'architecture': exp_info['architecture'],
                'parameters': results['architecture_params'],
                'training_time_minutes': results['training_time'],
                'results_by_lead_time': results['lead_times']
            }

    return summary


def main():
    """Main analysis function"""

    log_dir = os.path.expanduser('~/ai_weather_ag/data/architecture_experiments/logs')
    output_dir = os.path.expanduser('~/ai_weather_ag/data/architecture_experiments')

    print("=" * 80)
    print("ANALYZING ARCHITECTURE EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"\nLog directory: {log_dir}")
    print("")

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        print("Please run experiments first: ./run_architecture_experiments.sh")
        return 1

    # Parse all results
    results_dict = {}

    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"Processing: {exp_info['name']}...")
        log_file = find_latest_log(log_dir, exp_name)

        if log_file:
            print(f"  Found log: {os.path.basename(log_file)}")
            results = parse_log_file(log_file)
            if results:
                results_dict[exp_name] = results
                print(f"  ✓ Results extracted")
            else:
                print(f"  ✗ Failed to parse results")
        else:
            print(f"  ⚠ No log file found")
        print("")

    if not results_dict:
        print("No results found. Please run experiments first.")
        return 1

    # Create report
    print("Generating comparison report...")
    report = create_comparison_report(results_dict)

    # Save report
    report_path = os.path.join(output_dir, 'ARCHITECTURE_COMPARISON_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    # Save JSON summary
    json_summary = create_json_summary(results_dict)
    json_path = os.path.join(output_dir, 'results_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)

    print(f"JSON summary saved to: {json_path}")

    # Display report
    print("\n" + report)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
