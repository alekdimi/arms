# ARMS: Antithetic-REINFORCE-Multi-Sample Gradient for Binary Variables
-----
This is the official code repository for ICML 2021 paper: ARMS: Antithetic-REINFORCE-Multi-Sample Gradient for Binary Variables
by Alek Dimitriev and Mingyuan Zhou.

Full paper: [https://arxiv.org/pdf/2105.14141.pdf](https://arxiv.org/pdf/2105.14141.pdf)

To run an experiment you can start with the below template:
```
python3 -m experiment_launcher \
    --dataset=omniglot \
    --logdir=../logs_test \
    --grad_type=arms \
    --encoder_type=nonlinear \
    --num_steps=1e6 \
    --num_pairs=3 \
    --demean_input \
    --initialize_with_bias \
```
Support datasets are Dynamic MNIST, Fashion MNIST, and Omniglot, with either a linear or nonlinear encoder/decoder pair. 
Supported gradients are ARM, DisARM, LOORF, RELAX and ARMS (Dirichlet copula) and ARMS_Normal (Gaussian copula). 

