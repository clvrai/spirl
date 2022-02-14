## Pre-Trained Models

We provide pre-trained skill prior and skill decoder models in the kitchen and maze environments.
Using these models you can skip the pre-training step and directly run SPiRL-based RL on the target task. 
(**Note**: we provide weights for the default [model with closed-loop decoder](spirl/configs/skill_prior_learning/kitchen/hierarchical_cl/README.md))

To download the pre-trained model weights from Google Drive via the command line, you can use the 
[gdown](https://github.com/wkentaro/gdown) package. Install it with:
```
pip install gdown
```

Then create the appropriate weight directory and download the pre-trained weights. 
For the kitchen model run the following commands in the command line:

```
cd ${EXP_DIR}
mkdir -p skill_prior_learning/kitchen/hierarchical_cl/weights
cd skill_prior_learning/kitchen/hierarchical_cl/weights
gdown https://drive.google.com/uc?id=1uL6YVTGis2ltJJmBdO0LfTowHGSxjNrM
```

Then navigate back to the repository and start RL training using the example command from the main README:
```
python3 spirl/rl/train.py --path=spirl/configs/hrl/kitchen/spirl_cl --seed=0 --prefix=SPIRL_kitchen_seed0
```

Analogously download the pre-trained weights for the maze environment via:
```
cd ${EXP_DIR}
mkdir -p skill_prior_learning/maze/hierarchical_cl/weights
cd skill_prior_learning/maze/hierarchical_cl/weights
gdown https://drive.google.com/uc?id=1ojZNimi986UYhUkuZr5LfXK-JF72TggG
```

If you prefer to manually download the weights from Google Drive, follow the links below:

|Environment        | Link         | Size |
|:------------- |:-------------|:-----|
| Kitchen | [https://drive.google.com/file/d/1uL6YVTGis2ltJJmBdO0LfTowHGSxjNrM/view?usp=sharing](https://drive.google.com/file/d/1uL6YVTGis2ltJJmBdO0LfTowHGSxjNrM/view?usp=sharing) | 4MB |
| Maze |[https://drive.google.com/file/d/1uL6YVTGis2ltJJmBdO0LfTowHGSxjNrM/view?usp=sharing](https://drive.google.com/file/d/1uL6YVTGis2ltJJmBdO0LfTowHGSxjNrM/view?usp=sharing)| 4MB|
