# Office Cleanup Environment

![](https://kpertsch.github.io/resources/office.gif)

The task for the robot in the office cleanup environment is to place different objects into their target containers. 
We collect task-agnostic training data by randomly placing objects into random containers. For the target task 
we then fix one required matching between objects and containers.

## Installation

To install the environment, clone [our fork of the Roboverse repo](https://github.com/VentusYue/roboverse) and run:
```
cd roboverse
pip3 install -r requirements.txt
pip3 install -e .
```

## Data

We provide a dataset of task-agnostic sequences for SPiRL pre-training. For download commands and instructions for generating
more data, see [here](../README.md).

## Usage

To train a SPiRL skill model on the Office dataset, run:
```
python3 spirl/train.py --path=spirl/configs/skill_prior_learning/office/hierarchical_cl --val_data_size=160
``` 

To train a SPiRL agent on the target task, run:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/office --seed=0 --prefix=SPIRL_office_seed0
```
