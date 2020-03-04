# Isospectralization in PyTorch

## Introduction

This is an implementation for the paper "Isospectralization, or how to hear shape, style, and correspondence [1]" in PyTorch.

## Set up the environment

We recommend to use `pipenv` for environment setup. Please make sure that `pipenv` is properly installed (if not, try `pip install pipenv`), then run `pipenv install` to install the dependencies.

## TODO

- [ ] Experiment complex shapes in ShapeNet
- [ ] Reproduce style transfer results
- [ ] Reproduce nonisometric shape matching results (need to implement another model, so perhaps leave this for last)
- [ ] Improve CLI (possibly using argparse)

## Reference

1. Luca Cosmo, Mikhail Panine, Arianna Rampini, Maks Ovsjanikov, Michael M. Bronstein, Emanuele Rodolà, "Isospectralization, or how to hear shape, style, and correspondence," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
