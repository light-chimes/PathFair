# PathFair

> This is the code implementation of
> paper [PathFair: Improve White-Box Fairness Testing With Activation Path Guidance]()

> The implementation of [NeuronFair](https://dl.acm.org/doi/abs/10.1145/3510003.3510123) is modified based on
> the [code](https://github.com/haibinzheng/NeuronFair) provided by the authors.

> The implementation of activation path calculation references the [code](https://github.com/Antimony5292/FairNeuron)
> provided by the authors of [FairNeuron](https://dl.acm.org/doi/abs/10.1145/3510003.3510087).

## Introduction

***

As a rapidly evolving AI technology, deep neural networks are becoming more and more integrated with human society.
However, as the process progressed, concerns about fairness problems which might hide in neural networks were raised.
Previous studies have proposed a metric called counterfactual fairness to measure the fairness of machine learning
models, and proposed some methods to generate individual discrimination instance pairs (IDIPs) to improve the fairness
of the model by retraining the model after correcting the discovered IDIPs. However, existing methods generally do not
consider the selection of appropriate input samples, which may result in the performance of the method not being
sufficiently developed. In this paper, we propose an activation path-guided sample ranking and selection method with an
adjustable sampling rate parameter, followed by experiments on NeuronFair with several common datasets and sensitive
attributes for two possible parameter values. The results show that, on average, PathFair can improve the performance of
NeuronFair. Although we only conduct experiments on NeuronFair, since this method is a pre-processing method for sorting
input samples, this method should be also applicable to other IDIP generation methods.

![](regnet.png)

## Quick Start

***

### Preparation

#### Dependency

* joblib~=1.1.1
* scikit-learn~=0.22
* tensorflow~=2.8.0
* numpy~=1.22.4
* scipy~=1.9.2
* six~=1.16.0
* aif360~=0.5.0

We recommend using Anaconda to create and manage python environment. Our python version is 3.8.

#### Dataset

Datasets are available on [GitHub](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data), and can be
preprocessed with AIF360.You can also run the script directly and follow AIF360's prompts to download and install raw
dataset files.

### Get Started

Run following shell command from **PathFair/src** directory to start a white-box fairness testing, changing the path may
cause some error.

```shell
# param config_path: the path of configuration file
python path_fair.py --config_path ../configs/census_race_actpath_front.cfg
```

### Custom Configuration

Configuration files are placed in PathFair/configs directory, where we have provided all configurations used in our
experiments for different datasets and their sensitive attributes. Here are options in configuration files, you can
change them according to your needs:

```shell
# experiment name, will prompt when program starts
exp_name=census_gender+relu5+actpath_front+path_recover

# dataset, one of census/bank/compas/meps, corresponds to Adult/Bank/COMPAS/MEPS in paper
dataset=census

# optional, name of sensitive attribute
sens_name=gender

# index of sensitive attribute, index start from 1
sensitive_param=9

# path of trained neural network model
model_path=../models/census/dnn/best.model

# NeuronFair cluster num, used to cluster samples before global generation stage
cluster_num=4

# NeuronFair maximum number of input samples in global generation stage
max_global=1000

# NeuronFair maximum number of input samples in local generation stage
max_local=1000

# NeuronFair maximum number of perturbations in global generation stage
max_iter=40

# NeuronFair the most biased layer
ReLU_name=ReLU5

# Pre-processing method for input samples, one of random/cluster/actpath/actpath_front, cluster corresponds to Pcluster in paper, actpath to Pact_top, and actpath_front to Pact_sample
data_preproc=actpath_front

# Sampling rate for proposed method Pact_sample, necessary if data_preproc is set to actpath_front 
sample_rate=0.5

# NeuronFair perturbation step size in local and global generation
perturbation_size=1

# Random seed
uni_seed=3607

# NeuronFair decay value of momentum in local generation stage
local_decay=0.05

# NeuronFair decay value of momentum in global generation stage
global_dacay=0.1

# NeuronFair percentage of biased neurons in the most biased layer
bias_alpha=0.6375
```

## Citation

***

If you find this code useful for your research, please consider citing:

To be continued.