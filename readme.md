# Tensorflow Multilabel Hierarchical Attention Networks

## About

This package implements the Hierarchical Attention Networks from [Yang et al. 2016](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).

Main modifications are **multilabel** output with `sigmoid_cross_entropy_with_logits` and the addition of dense layers before the classification layers. Reason for this is to give the classifier more abstraction power throught these layers instead of relying on one single neuron to make sense of the document representation output from the 2 layers of bi-directionnal RNNs.

Code style is [Black](https://github.com/ambv/black).

## Install

## File structure

## Usage

---
## Credits
