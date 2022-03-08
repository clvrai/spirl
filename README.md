# Accelerating Reinforcement Learning with Learned Skill Priors
#### [[Project Website]](https://clvrai.github.io/spirl/) [[Paper]](https://arxiv.org/abs/2010.11944)

[Karl Pertsch](https://kpertsch.github.io/)<sup>1</sup>, [Youngwoon Lee](https://youngwoon.github.io/)<sup>1</sup>, 
[Joseph Lim](https://www.clvrai.com/)<sup>1</sup>

<sup>1</sup>CLVR Lab, University of Southern California 

<a href="https://clvrai.github.io/spirl/">
<p align="center">
<img src="docs/resources/spirl_teaser.png" width="800">
</p>
</img></a>

This is the official PyTorch implementation of the paper "**Accelerating Reinforcement Learning with Learned Skill Priors**"
(CoRL 2020).

## Updates
- **[Feb 2022]**: added [pre-trained models](spirl/data/pretrained_models.md) for kitchen and maze environments
- **[Jul 2021]**: added robotic office cleanup environment 
(see [details & installation here](spirl/data/office/README.md))
- **[Apr 2021]**: extended improved SPiRL version to support image-based observations 
(see [example commands](spirl/configs/skill_prior_learning/block_stacking/hierarchical_cl/README.md))
- **[Mar 2021]**: added an improved version of SPiRL with closed-loop skill decoder 
(see [example commands](spirl/configs/skill_prior_learning/kitchen/hierarchical_cl/README.md))

## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04

## Installation Instructions

Create a virtual environment and install all required packages.
```
cd spirl
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies and package
pip3 install -r requirements.txt
pip3 install -e .
```

Set the environment variables that specify the root experiment and data directories. For example: 
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

Finally, install **our fork** of the [D4RL benchmark](https://github.com/kpertsch/d4rl) repository by following its installation instructions.
It will provide both, the kitchen environment as well as the training data for the skill prior model in kitchen and maze environment.

## Example Commands
All results will be written to [WandB](https://www.wandb.com/). Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train.py](spirl/train.py) and
[rl/train.py](spirl/rl/train.py) to match your account.

To train a skill prior model for the kitchen environment, run:
```
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size=160
```
**Note**: You can skip this step by downloading our pre-trained skill prior models -- see [instructions here](spirl/data/pretrained_models.md).

For training a SPIRL agent on the kitchen environment using the pre-trained skill prior from above, run:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl --seed=0 --prefix=SPIRL_kitchen_seed0
```

In both commands, `kitchen` can be replaced with `maze / block_stacking` to run on the respective environment. Before training models
on these environments, the corresponding datasets need to be downloaded (the kitchen dataset gets downloaded automatically) 
-- download links are provided below.
Additional commands for training baseline models / agents are also provided below. 

### Baseline Commands

- Train **Single-step action prior**:
```
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/kitchen/flat --val_data_size=160
```

- Run **Vanilla SAC**:
```
python3 spirl/rl/train.py --path=spirl/configs/rl/kitchen/SAC --seed=0 --prefix=SAC_kitchen_seed0
```

- Run **SAC w/ single-step action prior**:
```
python3 spirl/rl/train.py --path=spirl/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=0 --prefix=flatPrior_kitchen_seed0
```

- Run **BC + finetune**:
```
python3 spirl/rl/train.py --path=spirl/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=0 --prefix=bcFinetune_kitchen_seed0
```

- Run **Skill Space Policy w/o prior**:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/no_prior/ --seed=0 --prefix=SSP_noPrior_kitchen_seed0
```

Again, all commands can be run on `maze / block stacking` by replacing `kitchen` with the respective environment in the paths
(after downloading the datasets).


## Starting to Modify the Code

### Modifying the hyperparameters
The default hyperparameters are defined in the respective model files, e.g. in [```skill_prior_mdl.py```](spirl/models/skill_prior_mdl.py#L47)
for the SPIRL model. Modifications to these parameters can be defined through the experiment config files (passed to the respective
command via the `--path` variable). For an example, see [```kitchen/hierarchical/conf.py```](spirl/configs/skill_prior_learning/kitchen/hierarchical/conf.py).


### Adding a new dataset for model training
All code that is dataset-specific should be placed in a corresponding subfolder in `spirl/data`. 
To add a data loader for a new dataset, the `Dataset` classes from [```data_loader.py```](spirl/components/data_loader.py) need to be subclassed
and the `__getitem__` function needs to be overwritten to load a single data sample. The output `dict` should include the following
keys:

```
dict({
    'states': (time, state_dim)                 # state sequence (for state-based prior inputs)
    'actions': (time, action_dim)               # action sequence (as skill input for training prior model)
    'images':  (time, channels, width, height)  # image sequence (for image-based prior inputs)
})
```

All datasets used with the codebase so far have been based on `HDF5` files. The `GlobalSplitDataset` provides functionality to read all
HDF5-files in a directory and split them in `train/val/test` based on percentages. The `VideoDataset` class provides
many functionalities for manipulating sequences, like randomly cropping subsequences, padding etc.

### Adding a new RL environment
To add a new RL environment, simply define a new environent class in `spirl/rl/envs` that inherits from the environment interface
in [```spirl/rl/components/environment.py```](spirl/rl/components/environment.py).


### Modifying the skill prior model architecture
Start by defining a model class in the `spirl/models` directory that inherits from the `BaseModel` or `SkillPriorMdl` class. 
The new model needs to define the architecture in the constructor (e.g. by overwriting the `build_network()` function), 
implement the forward pass and loss functions,
as well as model-specific logging functionality if desired. For an example, see [```spirl/models/skill_prior_mdl.py```](spirl/models/skill_prior_mdl.py).

Note, that most basic architecture components (MLPs, CNNs, LSTMs, Flow models etc) are defined in `spirl/modules` and can be 
conveniently reused for easy architecture definitions. Below are some links to the most important classes.

|Component        | File         | Description |
|:------------- |:-------------|:-------------|
| MLP | [```Predictor```](spirl/modules/subnetworks.py#L33) | Basic N-layer fully-connected network. Defines number of inputs, outputs, layers and hidden units. |
| CNN-Encoder | [```ConvEncoder```](spirl/modules/subnetworks.py#L66) | Convolutional encoder, number of layers determined by input dimensionality (resolution halved per layer). Number of channels doubles per layer. Returns encoded vector + skip activations. |
| CNN-Decoder | [```ConvDecoder```](spirl/modules/subnetworks.py#L145) | Mirrors architecture of conv. encoder. Can take skip connections as input, also versions that copy pixels etc. |
| Processing-LSTM | [```BaseProcessingLSTM```](spirl/modules/recurrent_modules.py#L70) | Basic N-layer LSTM for processing an input sequence. Produces one output per timestep, number of layers / hidden size configurable.|
| Prediction-LSTM | [```RecurrentPredictor```](spirl/modules/recurrent_modules.py#L241) | Same as processing LSTM, but for autoregressive prediction. |
| Mixture-Density Network | [```MDN```](spirl/modules/mdn.py#L10) | MLP that outputs GMM distribution. |
| Normalizing Flow Model | [```NormalizingFlowModel```](spirl/modules/flow_models.py#L9) | Implements normalizing flow model that stacks multiple flow blocks. Implementation for RealNVP block provided. |

### Adding a new RL algorithm
The core RL algorithms are implemented within the `Agent` class. For adding a new algorithm, a new file needs to be created in
`spirl/rl/agents` and [```BaseAgent```](spirl/rl/components/agent.py#L19) needs to be subclassed. In particular, any required
networks (actor, critic etc) need to be constructed and the `update(...)` function needs to be overwritten. For an example, 
see the SAC implementation in [```SACAgent```](spirl/rl/agents/ac_agent.py#L67).

The main SPIRL skill prior regularized RL algorithm is implemented in [```ActionPriorSACAgent```](spirl/rl/agents/prior_sac_agent.py#L12).


## Detailed Code Structure Overview
```
spirl
  |- components            # reusable infrastructure for model training
  |    |- base_model.py    # basic model class that all models inherit from
  |    |- checkpointer.py  # handles storing + loading of model checkpoints
  |    |- data_loader.py   # basic dataset classes, new datasets need to inherit from here
  |    |- evaluator.py     # defines basic evaluation routines, eg top-of-N evaluation, + eval logging
  |    |- logger.py        # implements core logging functionality using tensorboardX
  |    |- params.py        # definition of command line params for model training
  |    |- trainer_base.py  # basic training utils used in main trainer file
  |
  |- configs               # all experiment configs should be placed here
  |    |- data_collect     # configs for data collection runs
  |    |- default_data_configs   # defines one default data config per dataset, e.g. state/action dim etc
  |    |- hrl              # configs for hierarchical downstream RL
  |    |- rl               # configs for non-hierarchical downstream RL
  |    |- skill_prior_learning   # configs for skill embedding and prior training (both hierarchical and flat)
  |
  |- data                  # any dataset-specific code (like data generation scripts, custom loaders etc)
  |- models                # holds all model classes that implement forward, loss, visualization
  |- modules               # reusable architecture components (like MLPs, CNNs, LSTMs, Flows etc)
  |- rl                    # all code related to RL
  |    |- agents           # implements core algorithms in agent classes, like SAC etc
  |    |- components       # reusable infrastructure for RL experiments
  |        |- agent.py     # basic agent and hierarchial agent classes - do not implement any specific RL algo
  |        |- critic.py    # basic critic implementations (eg MLP-based critic)
  |        |- environment.py    # defines environment interface, basic gym env
  |        |- normalization.py  # observation normalization classes, only optional
  |        |- params.py    # definition of command line params for RL training
  |        |- policy.py    # basic policy interface definition
  |        |- replay_buffer.py  # simple numpy-array replay buffer, uniform sampling and versions
  |        |- sampler.py   # rollout sampler for collecting experience, for flat and hierarchical agents
  |    |- envs             # all custom RL environments should be defined here
  |    |- policies         # policy implementations go here, MLP-policy and RandomAction are implemented
  |    |- utils            # utilities for RL code like MPI, WandB related code
  |    |- train.py         # main RL training script, builds all components + runs training
  |
  |- utils                 # general utilities, pytorch / visualization utilities etc
  |- train.py              # main model training script, builds all components + runs training loop and logging
```

The general philosophy is that each new experiment gets a new config file that captures all hyperparameters etc. so that experiments
themselves are version controllable.

## Datasets

|Dataset        | Link         | Size |
|:------------- |:-------------|:-----|
| Maze | [https://drive.google.com/file/d/1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ/view?usp=sharing](https://drive.google.com/file/d/1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ/view?usp=sharing) | 12GB |
| Block Stacking |[https://drive.google.com/file/d/1VobNYJQw_Uwax0kbFG7KOXTgv6ja2s1M/view?usp=sharing](https://drive.google.com/file/d/1VobNYJQw_Uwax0kbFG7KOXTgv6ja2s1M/view?usp=sharing)| 11GB|
| Office Cleanup | [https://drive.google.com/file/d/1yNsTZkefMMvdbIBe-dTHJxgPIRXyxzb7/view?usp=sharing](https://drive.google.com/file/d/1yNsTZkefMMvdbIBe-dTHJxgPIRXyxzb7/view?usp=sharing)| 170MB |

You can download the datasets used for the experiments in the paper with the links above. 
To download the data via the command line, see example commands [here](spirl/data/).

If you want to generate more data 
or make other modifications to the data generating procedure, we provide instructions for regenerating the 
`maze`, `block stacking` and `office` datasets [here](spirl/data/).


## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{pertsch2020spirl,
    title={Accelerating Reinforcement Learning with Learned Skill Priors},
    author={Karl Pertsch and Youngwoon Lee and Joseph J. Lim},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2020},
}
```

## Acknowledgements
The model architecture and training code builds on a code base which we jointly developed with [Oleh Rybkin](https://www.seas.upenn.edu/~oleh/) for our previous project on [hierarchial prediction](https://github.com/orybkin/video-gcp).

We also published many of the utils / architectural building blocks in a stand-alone package for easy import into your 
own research projects: check out the [blox](https://github.com/orybkin/blox-nn) python module. 


## Troubleshooting

### Missing key 'completed_tasks' in Kitchen environment
Please make sure to install [our fork](https://github.com/kpertsch/d4rl) of the D4RL repository, **not** the original D4RL repository. We made a few small changes to the interface, which e.g. allow us to log the reward contributions for each of the subtasks separately.




