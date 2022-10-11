#  Can Push-forward Generative Models Fit Multimodal Distributions?

This repository contains the code for reproducing the main experiments of the paper [Can Push-forward Generative Models Fit Multimodal Distributions?](https://arxiv.org/abs/2206.14476)



If using this code, please cite the paper:

<pre><code> @article{salmona2022can,
  title={Can Push-forward Generative Models Fit Multimodal Distributions?},
  author={Salmona, Antoine and de Bortoli, Valentin and Delon, Julie and Desolneux, Agn{\`e}s},
  journal={arXiv preprint arXiv:2206.14476},
  year={2022}
}  </code></pre>

## Contributors

- Antoine Salmona
- Valentin de Bortoli
- Julie Delon
- Agnès Desolneux


## Installation

This project can be installed by running:

    conda env create -f conda.yaml

    conda activate pushforward


## Main dependencies

the main dependencies are: 

    - matplotlib==3.4.2

    - numpy==1.20.2

    - POT==0.8.0

    - scipy==1.6.2

    - torch==1.8.1

    - torchvision==0.9.1

    - tqdm==4.61.0


## How to use this code?

The main experiments of the paper can be reproduced using the three runners 
runner_dim1.py, runner_GM_MNIST.py, runner_subset_MNIST.py. 

### Univariate case

runner_dim1.py takes two mandatory arguments: 


    --model: 'VAE', 'GAN', and 'SGM' i.e. the type of model to train

    --expe: 0 for the main experiment (Figure 1 and 2 in the paper), 

            1 for the experiment on the GAN with Gradient Penalty (Figure 3 in the paper),

            2 for the experiment on the numbers of layers (Figure 4 in the paper).

other arguments: 


    --n_runs: number of runs of the experiments (default 1 here, 10 in the paper),

    --n_epochs: number of training epochs,

    --batch_size,

    --lr: learning rate,

    --sigma_ratio,--L,--n_step_each,--step_lr: hyperparameters of the SGM.

To reproduce the main experiment, run: 

    python3 runner_dim1.py --model=VAE --expe=0 --n_runs=10 

    python3 runner_dim1.py --model=GAN --expe=0 --n_runs=10 

    python3 runner_dim1.py --model=SGM --expe=0 --n_runs=10 


To reproduce the experiment on the GAN with gradient penalty, run: 

    python3 runner_dim1.py --model=GAN --expe=1


To reproduce the experiment on the numbers of layers, run: 

    python3 runner_dim1.py --model=GAN --expe=2 --n_runs=10

    python3 runner_dim1.py --model=VAE --expe=2 --n_runs=10

The histograms of the generated distributions and the data to reproduce the figures 
of the paper are then in the folder results_dim1. 

### Gaussian mixture on MNIST 

runner_GM_MNIST.py takes one mandatory argument: 

    --model: 'VAE', 'GAN', and 'SGM', or 'data', i.e the type of model.

other arguments: 

    --n_epochs: number of training epochs,

    --batch_size,

    --lr: learning rate,

    --sigma_ratio,--L,--n_step_each,--step_lr: hyperparameters of the SGM,   

    --digita and --digitb: the two subsets of digits of MNIST to select (by default 3 and 7),

    --batch_gen: batch size for generation,

    --n_gen: number of generated samples for the histogram,

    --n_latent: dimension of the latent space for the VAE and the GAN,

    --n_feature: numbers of channels for the VAE and the GAN, and x2 for the SGM (by default 32 for VAE and GAN, 64 for the SGM).

To reproduce Figure 6 top, run: 

    python3 runner_GM_MNIST.py --model=data

    python3 runner_GM_MNIST.py --model=VAE

    python3 runner_GM_MNIST.py --model=GAN

    python3 runner_GM_MNIST.py --model=SGM

The generated histograms, the generated data and the weights of the networks are then in the folder results_MNIST_GM. 



### Subset of MNIST 

runner_subset_MNIST.py takes one mandatory argument: 

    --model: 'VAE', 'GAN', and 'SGM', or 'data', i.e the type of model.

other arguments: 

    --n_epochs: number of training epochs,

    --batch_size,

    --lr: learning rate,

    --sigma_ratio,--L,--n_step_each,--step_lr: hyperparameters of the SGM,   

    --digita and --digitb: the two subsets of digits of MNIST to select (by default 3 and 7),

    --batch_gen: batch size for generation,

    --n_gen: number of generated samples for the histogram,

    --n_latent: dimension of the latent space for the VAE and the GAN,

    --n_feature: numbers of channels for the VAE and the GAN, and /4 for the SGM (by default 256 for VAE and GAN, 64 for the SGM),

    --self_attn: if True, add self attention layers to the score network. 

To reproduce Figure 6 bottom, run: 

    python3 runner_subset_MNIST.py --model=data

    python3 runner_subset_MNIST.py --model=VAE

    python3 runner_subset_MNIST.py --model=GAN

    python3 runner_subset_MNIST.py --model=SGM

The generated histograms, the generated data and the weights of the networks are then in the folder results_MNIST_subset. 
The code has been developped/tested on Python 3.8.10, Pytorch 1.8.1, and Cuda 10.1 with two NVIDIA Titan Xp. 
