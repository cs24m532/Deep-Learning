# VGG6 CIFAR-10 experiments (CS6886W Assignment 1)

### Prerequisites
torch>=1.12.0
torchvision>=0.13.0
tqdm
numpy
matplotlib
wandb
scikit-learn

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Single Run
```bash
python train.py --activation gelu --optimizer sgd --lr 0.01 --batch_size 128 --epochs 30
```

## WandB Sweep
```bash
wandb login
wandb sweep sweep.yaml
wandb agent <SWEEP_ID>
```

## Output
Models in ./checkpoints, plots and metrics in W&B.
