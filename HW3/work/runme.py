"""
Run neural_networks with 4 hyperparameter configs to produce grading JSON files.
"""
import sys
sys.path.insert(0, '.')
from neural_networks import main
import json

configs = [
    {'learning_rate': 0.01, 'momentum': 0.0, 'dropout': 0.0, 'activation': 'relu', 'n_hidden': 100, 'epochs': 10},
    {'learning_rate': 0.01, 'momentum': 0.0, 'dropout': 0.0, 'activation': 'tanh', 'n_hidden': 100, 'epochs': 10},
    {'learning_rate': 0.01, 'momentum': 0.9, 'dropout': 0.25, 'activation': 'tanh', 'n_hidden': 100, 'epochs': 10},
    {'learning_rate': 0.01, 'momentum': 0.9, 'dropout': 0.5, 'activation': 'relu', 'n_hidden': 100, 'epochs': 10},
]
names = [
    'MLP_lr0.01_m0.0_d0.0_arelu.json',
    'MLP_lr0.01_m0.0_d0.0_atanh.json',
    'MLP_lr0.01_m0.9_d0.25_atanh.json',
    'MLP_lr0.01_m0.9_d0.5_arelu.json',
]
for cfg, name in zip(configs, names):
    print(f'\n--- Running {name} ---')
    hist = main(cfg)
    with open(name, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f'Saved {name}')
