This is the code implementation of Reinforcement Learning with Adaptive Regularization (RL-AR) from the paper '*Reinforcement Learning with Adaptive Regularization for Safe Control of Critical Systems*' (Accepted by NeurIPS 2024).

## Running the code

The requirements are given in [requirements.txt](/requirements.txt). The code has been tested on Ubuntu 22.04.3 LTS, with Intel(R) Core(TM) i7-13850HX CPU and NVIDIA RTX 3500 Ada GPU with CUDA V12.1.66. The python version used is `python==3.12.5`.

* To see RL-AR training records using tensorboard, run the following command:
    * ``tensorboard --logdir=runs``

* For easy validation using trained models: [validation.ipynb](/validation.ipynb).  

* To train RL-AR using pretrained beta network:
    1. train RL-AR: [train_RLAR.ipynb](/train_RLAR.ipynb).

* To train RL-AR from scratch:  
    1. pretrain beta network: [pretrain_beta.ipynb](/pretrain_beta.ipynb). The training automatically terminates when $\beta\geq0.999$ for all 512 sampled states.
    2. train RL-AR: [train_RLAR.ipynb](/train_RLAR.ipynb).

## Credit

The implementation of the MPC agent relies on [do-mpc](https://github.com/do-mpc/do-mpc), [CasADi](https://web.casadi.org/), and [IPOPT](https://coin-or.github.io/Ipopt/).

If you find the repo useful, please cite us:

```
@article{tian2024reinforcement,
  title={Reinforcement Learning with Adaptive Control Regularization for Safe Control of Critical Systems},
  author={Tian, Haozhe and Hamedmoghadam, Homayoun and Shorten, Robert and Ferraro, Pietro},
  journal={arXiv preprint arXiv:2404.15199},
  year={2024}
}
```