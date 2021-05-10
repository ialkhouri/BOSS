# BOSS: Bidirectional One-Shot Synthesis of Adversarial Examples

This repository is the official implementation of our submitted paper to NIPS 2021. 

## Requirements

To install requirements, use file BOSS_conda_env.yml

## Execution 

- To reproduce results of figure 1, we use "GenNNs_BOSS_MNIST_digits.py", "GenNNs_BOSS_MNIST_fashion_HCs.py", "GenNNs_cifar_10.py", "GenNNs_BOSS_GTSRB.py", and "GenNNs_BOSS_COVID_19.py".

- For BOSS-C, we use "GenNNs_MNIST_digits_confide_reduction.py".

- For BOSS-U and BOSS-B, we use "GenNNs_MNIST_digits_boudary_examples.py".
 
- For BOSS-T, we use "GenNNs_MNIST_digits_targeted_attack.py"

- For CW attack, we use the implementation of the original paper located at "https://github.com/carlini/nn_robust_attacks".

- For NewtonFool, we use the implemenation of ART at "https://adversarial-robustness-toolbox.readthedocs.io/en/stable/modules/attacks/evasion.html#newtonfool"
