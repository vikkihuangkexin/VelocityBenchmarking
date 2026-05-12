# NeuroVelo

## Installation

```bash
pip install neurovelo
```

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers.

## Output

The velocity results are stored in the 'spliced_velocity' layer of the output h5ad file.

## Parameters

- `--data_dir`: Input h5ad data file path. Default: ./test.h5ad
- `--save_dir`: Result saving directory. Default: ./test
- `--n_ode_hidden`: Number of hidden units in ODE network. Default: 100
- `--n_vae_hidden`: Number of hidden units in VAE network. Default: 100
- `--n_latent`: Number of latent dimensions. Default: 50
- `--batch_size`: Batch size for training. Default: 100
- `--nepoch`: Number of training epochs. Default: 100
- `--simulate`: Whether the data is simulation data. If true, adjusts preprocessing parameters and sets X_umap from X_dimred if available.

## Usage

```bash
python neurovelo.py --data_dir data.h5ad --save_dir results --n_ode_hidden 150 --n_vae_hidden 150 --n_latent 40 --batch_size 50 --nepoch 200 --simulate
```

For more details, see https://github.com/idriskb/NeuroVelo
