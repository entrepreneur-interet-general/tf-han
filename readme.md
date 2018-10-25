# Tensorflow Multilabel Hierarchical Attention Networks

## About

This package implements the Hierarchical Attention Networks from [Yang et al. 2016](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).

Main modifications are **multilabel** output with `sigmoid_cross_entropy_with_logits` and the addition of dense layers before the classification layers. Reason for this is to give the classifier more abstraction power throught these layers instead of relying on one single neuron to make sense of the document representation output from the 2 layers of bi-directionnal RNNs.

Code style is [Black](https://github.com/ambv/black).

## Install

Main dependencies are `python (3.6)`, `tensorflow (1.7)`, `pandas`, `sklearn` and `PyYAML`.

```bash
$ git clone git@githib.com/vict0rsch/tf-han.git
$ cd tf-han
$ pip install -r requirements.txt
```


## Usage

### Outline
The idea here is to split the work between several calsses:

A `model` is the set of tensorflow operations that transform and `input_tensor` into the `output_tensor` (no shit)

A `trainer` trains (and validate) a `model`: it takes care of getting the data, computing the loss, optimizing the model, running the validation steps, saving the `graph` etc.

An `experiment` runs various trainers with various hyper parameters.

All these classes have an `hp` attribute which represents their hyperparameters.

### Available classes


### Configuring an experiment

All (hyper) parameters can be defined in a `yaml` configuration file. List of parameters and their values can be found there: [**parameters.md**](/parameters.md).

A default configuration file is provided -> [default_conf.yaml](/default_conf.yaml).

The idea is to have two separate sets of parameters:

* Fixed parameters like the source of data, the type of `trainer` or `model` or anything you want to keep constant across trainers during an experiment
* Randomizable parameters like the `batch_size` or `layers` or `optimizer` or `learning_rate` or anything you want to change between trainers

#### Randomizable parameters

They can be randomized in a few ways:

* From a `range` or from a `list` of values
* Sampled according a `uniform`, `normal` or `deterministic` (in the order of the range or list) distribution.

### Data format

### Running an Experiment
```
$ pwd
path/to/tf-han
$ ls
readme.md        requirements.txt test.py          text_classifier
$ ipython
```

```python
>>> import text_classifier as tcl
>>> exp = tcl.Experiment(conf_path='path/to/conf.yaml')
>>> exp.run(10)
```




## File structure
---
## Credits
