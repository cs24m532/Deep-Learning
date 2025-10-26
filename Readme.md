# VGG6 CIFAR-10 experiments
# Copy all the files in your machines.Open command prompt and run from there using python

## Seed value : 42
## Setup and install required prerequisites
```bash
python -m venv venv
source venv/bin/activate
pip install -r prerequisites.txt
```

## Single Run
```bash
python train.py --activation selu --optimizer adam --lr 0.001 --batch_size 64 --epochs 20
```

## WandB Sweep [to run in multiple instance]
```bash
python -m wandb login
python -m wandb sweep sweep.yaml
python -m wandb agent <SWEEP_ID>
```
## Output
Models in ./checkpoints, plots and metrics in W&B.

