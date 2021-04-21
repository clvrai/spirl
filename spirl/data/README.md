## Downloading Datasets via the Command Line

To download the dataset files from Google Drive via the command line, you can use the 
[gdown](https://github.com/wkentaro/gdown) package. Install it with:
```
pip install gdown
```

Then navigate to the folder you want to download the data to and run the following commands:
```
# Download Maze Dataset
gdown https://drive.google.com/uc?id=1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ

# Download Block Stacking Dataset
gdown https://drive.google.com/uc?id=1VobNYJQw_Uwax0kbFG7KOXTgv6ja2s1M
``` 

## Re-Generating Datasets

### Maze Dataset
To regenerate the maze dataset, our fork of the [D4RL repo](https://github.com/kpertsch/d4rl) needs to be cloned and installed.
It includes the script used to generate the maze dataset. Specifically, new data can be created by running:
```
cd d4rl
python3 scripts/generate_randMaze2d_datasets.py --render --agent_centric --save_images --data_dir=path_to_your_dir
```
Optionally, an argument `--batch_idx` allows to automatically generate a subfolder in `data_dir` with the batch index, 
so that multiple data generation scripts with different batch indices can be run in parallel
for accelerated data generation.

The number of trajectories that are getting generated can be controlled through the argument `--num_samples`; the size
of the randomly generated training mazes can be changed with `--rand_maze_size`. For a full list of all arguments, see
[```scripts/generate_randMaze2d_datasets.py```](https://github.com/kpertsch/d4rl/scripts/generate_randMaze2d_datasets.py#L72).


### Block Stacking Dataset
To regenerate the block stacking dataset we can use the config provided in [```spirl/configs/data_collect/block_stacking```](spirl/configs/data_collect/block_stacking/conf.py).
To start generation, run:
```
python3 spirl/rl/train.py --path=spirl/configs/data_collect/block_stacking --mode=rollout --n_val_samples=2e5 --seed=42 --data_dir=path_to_your_dir
```
If you want to run multiple data generation jobs in parallel, make sure to change the seed and set a different target 
data directory.