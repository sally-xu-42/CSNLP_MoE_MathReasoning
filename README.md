# Applying Mixture of Experts Technique on Small-Size Language Models for Multi-Step Mathematical Reasoning Problems

## on euler server
### conda
```conda create -n csnlp_conda python=3.8```
### Or venv
```
mkdir venvs
python3 -m venv venvs/csnlp_venv
source venvs/csnlp_venv/bin/activate
```
### euler module
```module load gcc/8.2.0 python_gpu/3.10.4 boost/1.74.0 eth_proxy```

```pip install -r requirements.txt```

### login wandb
```
wandb login
```

### submit euler job
```sbatch jobs/train```
