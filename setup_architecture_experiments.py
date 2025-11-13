#!/usr/bin/env python3
"""
Architecture Experiment Script for India Region (6x6)

Compares 4 architectures:
- MLP_Deep: Deep MLP with 6 hidden layers
- MLP_Wide: Wide MLP with 2048 neurons per layer
- UNet_Light: Lightweight UNet with 64 base channels
- UNet_Deep: Deep UNet with 4 encoder/decoder levels

Variables:
- Input: 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind,
         temperature_1000hPa, specific_humidity_1000hPa, geopotential_1000hPa
- Output: 2m_temperature

Model: Pangu
Region: India (6x6 subregion)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
EXPERIMENTS = {
    'mlp_deep': {
        'architecture': 'mlp',
        'hidden_dim': 1024,
        'num_hidden_layers': 6,
        'dropout_rate': 0.25,
        'description': 'Deep MLP with 6 hidden layers (1024 neurons each)'
    },
    'mlp_wide': {
        'architecture': 'mlp',
        'hidden_dim': 2048,
        'num_hidden_layers': 3,
        'dropout_rate': 0.3,
        'description': 'Wide MLP with 3 hidden layers (2048 neurons each)'
    },
    'unet_light': {
        'architecture': 'unet',
        'hidden_dim': 64,
        'dropout_rate': 0.1,
        'description': 'Lightweight UNet with 64 base channels'
    },
    'unet_deep': {
        'architecture': 'unet',
        'hidden_dim': 128,
        'dropout_rate': 0.15,
        'description': 'Deep UNet with 128 base channels'
    }
}

# Common parameters for all experiments
COMMON_PARAMS = {
    'region': 'india',
    'subregion': '6x6',
    'model_name': 'pangu',
    'ground_truth_source': '',
    'training_vars': [
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'temperature_1000hPa',
        'specific_humidity_1000hPa',
        'geopotential_1000hPa'
    ],
    'output_vars': ['2m_temperature'],
    'lead_time_hours': [24, 72, 144],  # 1, 3, and 6 day forecasts
    'train_start': '2020-01-01',
    'train_end': '2020-12-31',
    'test_start': '2021-01-01',
    'test_end': '2021-06-30',  # 6 months for faster testing
    'data_dir': os.path.expanduser('~/ai_weather_ag/data/raw'),
    'output_dir': os.path.expanduser('~/ai_weather_ag/data/architecture_experiments')
}


def run_experiment(exp_name, exp_config):
    """Run a single architecture experiment"""

    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_name.upper()}")
    print(f"Description: {exp_config['description']}")
    print("="*80)

    # Build command
    cmd_parts = [
        'python3 finetuning/finetune.py',
        f'--region={COMMON_PARAMS["region"]}',
        f'--subregion={COMMON_PARAMS["subregion"]}',
        f'--model_name={COMMON_PARAMS["model_name"]}',
        f'--ground_truth_source={COMMON_PARAMS["ground_truth_source"]}',
        f'--train_start={COMMON_PARAMS["train_start"]}',
        f'--train_end={COMMON_PARAMS["train_end"]}',
        f'--test_start={COMMON_PARAMS["test_start"]}',
        f'--test_end={COMMON_PARAMS["test_end"]}',
        f'--data_dir={COMMON_PARAMS["data_dir"]}',
        f'--output_dir={COMMON_PARAMS["output_dir"]}',
        f'--nn_architecture={exp_config["architecture"]}',
        f'--training_vars {" ".join(COMMON_PARAMS["training_vars"])}',
        f'--output_vars {" ".join(COMMON_PARAMS["output_vars"])}',
        f'--lead_time_hours {" ".join(map(str, COMMON_PARAMS["lead_time_hours"]))}',
    ]

    # Add architecture-specific parameters
    if exp_config['architecture'] == 'mlp':
        # Note: These will need to be added as command-line arguments or modified in the code
        print(f"\nArchitecture: MLP")
        print(f"  Hidden dim: {exp_config['hidden_dim']}")
        print(f"  Num layers: {exp_config['num_hidden_layers']}")
        print(f"  Dropout: {exp_config['dropout_rate']}")
    else:
        print(f"\nArchitecture: UNet")
        print(f"  Hidden dim: {exp_config['hidden_dim']}")
        print(f"  Dropout: {exp_config['dropout_rate']}")

    cmd = ' \\\n    '.join(cmd_parts)

    print(f"\nCommand:\n{cmd}")
    print("\n" + "-"*80)

    # Create output directory
    os.makedirs(COMMON_PARAMS['output_dir'], exist_ok=True)

    # Save experiment config
    config_path = os.path.join(COMMON_PARAMS['output_dir'], f'{exp_name}_config.json')
    config_data = {
        'experiment_name': exp_name,
        'description': exp_config['description'],
        'architecture': exp_config['architecture'],
        'parameters': exp_config,
        'common_params': COMMON_PARAMS,
        'timestamp': datetime.now().isoformat()
    }

    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    return cmd


def create_experiment_runner():
    """Create a shell script to run all experiments"""

    script_path = 'run_architecture_experiments.sh'

    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Architecture Experiment Runner\n')
        f.write('# Compares MLP and UNet architectures for India region\n\n')
        f.write(f'# Generated: {datetime.now().isoformat()}\n\n')

        for exp_name, exp_config in EXPERIMENTS.items():
            f.write(f'\n# {"-"*76}\n')
            f.write(f'# Experiment: {exp_name}\n')
            f.write(f'# {exp_config["description"]}\n')
            f.write(f'# {"-"*76}\n\n')

            cmd = run_experiment(exp_name, exp_config)
            f.write(cmd + '\n\n')
            f.write('if [ $? -ne 0 ]; then\n')
            f.write(f'    echo "ERROR: Experiment {exp_name} failed"\n')
            f.write('    exit 1\n')
            f.write('fi\n\n')
            f.write(f'echo "✓ Experiment {exp_name} completed successfully"\n')
            f.write('echo ""\n\n')

    # Make executable
    os.chmod(script_path, 0o755)

    print(f"\n{'='*80}")
    print(f"Experiment runner script created: {script_path}")
    print(f"{'='*80}")

    return script_path


def create_manual_experiment_instructions():
    """Create instructions for manually updating architecture parameters"""

    instructions = """
# Manual Architecture Experiment Instructions

The finetune.py script currently uses hardcoded architecture parameters.
To run different architecture experiments, you need to temporarily modify
the architecture initialization in finetune.py.

## Current Architecture Initialization Locations:

### For MLP (around line 1009-1016):
```python
model = SimpleMLP(input_dim = input_dim,
                  hidden_dim = 1024,  # MODIFY THIS
                  output_dim = output_dim,
                  num_hidden_layers= 4,  # MODIFY THIS
                  n_lead_times=n_lead_times,
                  lead_time_embedding_dim=4,
                  dropout_rate=0.2477893381  # MODIFY THIS
                  ).to(device)
```

### For UNet (around line 1003-1005):
```python
model = UNet(input_dim, 32,  # MODIFY hidden_dim HERE
             output_dim, n_lat=n_lat, n_lon=n_lon,
             n_input_vars=n_training_vars, n_output_vars=n_output_vars,
             n_lead_times=n_lead_times).to(device)
```

## Experiments to Run:

### 1. MLP_Deep
Modify SimpleMLP initialization to:
- hidden_dim = 1024
- num_hidden_layers = 6
- dropout_rate = 0.25

Run: python3 finetuning/finetune.py --nn_architecture=mlp --region=india --subregion=6x6 ...

### 2. MLP_Wide
Modify SimpleMLP initialization to:
- hidden_dim = 2048
- num_hidden_layers = 3
- dropout_rate = 0.3

Run: python3 finetuning/finetune.py --nn_architecture=mlp --region=india --subregion=6x6 ...

### 3. UNet_Light
Modify UNet initialization to:
- hidden_dim = 64
- dropout_rate = 0.1

Run: python3 finetuning/finetune.py --nn_architecture=unet --region=india --subregion=6x6 ...

### 4. UNet_Deep
Modify UNet initialization to:
- hidden_dim = 128
- dropout_rate = 0.15

Run: python3 finetuning/finetune.py --nn_architecture=unet --region=india --subregion=6x6 ...

## Full Command Template:

```bash
python3 finetuning/finetune.py \\
    --region=india \\
    --subregion=6x6 \\
    --model_name=pangu \\
    --nn_architecture=mlp \\  # or unet
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \\
    --output_vars 2m_temperature \\
    --lead_time_hours 24 72 144 \\
    --train_start=2020-01-01 --train_end=2020-12-31 \\
    --test_start=2021-01-01 --test_end=2021-06-30 \\
    --data_dir=~/ai_weather_ag/data/raw \\
    --output_dir=~/ai_weather_ag/data/architecture_experiments
```

## Analyzing Results:

After running each experiment, the output will include:
- MSE for each lead time (24h, 72h, 144h)
- Training time
- Model performance vs original Pangu forecast

Look for lines like:
```
Lead time 24h - MSE original: X.XXXXXX, MSE corrected: Y.YYYYYY
Lead time 72h - MSE original: X.XXXXXX, MSE corrected: Y.YYYYYY
Lead time 144h - MSE original: X.XXXXXX, MSE corrected: Y.YYYYYY
```

Calculate RMSE improvement:
RMSE_original = sqrt(MSE_original)
RMSE_corrected = sqrt(MSE_corrected)
Improvement = (RMSE_original - RMSE_corrected) / RMSE_original * 100%
"""

    instructions_path = 'ARCHITECTURE_EXPERIMENT_INSTRUCTIONS.md'
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(f"Manual instructions saved to: {instructions_path}")

    return instructions_path


if __name__ == '__main__':
    print("="*80)
    print("ARCHITECTURE EXPERIMENT SETUP")
    print("India Region (6x6 subregion)")
    print("="*80)

    print("\nExperiments to run:")
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"  - {exp_name}: {exp_config['description']}")

    print(f"\nCommon parameters:")
    print(f"  Region: {COMMON_PARAMS['region']} ({COMMON_PARAMS['subregion']})")
    print(f"  Model: {COMMON_PARAMS['model_name']}")
    print(f"  Training: {COMMON_PARAMS['train_start']} to {COMMON_PARAMS['train_end']}")
    print(f"  Testing: {COMMON_PARAMS['test_start']} to {COMMON_PARAMS['test_end']}")
    print(f"  Input variables: {', '.join(COMMON_PARAMS['training_vars'])}")
    print(f"  Output variable: {', '.join(COMMON_PARAMS['output_vars'])}")
    print(f"  Lead times: {COMMON_PARAMS['lead_time_hours']} hours")

    # Create experiment runner
    create_experiment_runner()

    # Create manual instructions
    create_manual_experiment_instructions()

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\nSince architecture parameters need to be modified in finetune.py,")
    print("please follow the instructions in: ARCHITECTURE_EXPERIMENT_INSTRUCTIONS.md")
    print("\nOr I can modify finetune.py to accept architecture parameters as")
    print("command-line arguments, making experiments easier to run.")
