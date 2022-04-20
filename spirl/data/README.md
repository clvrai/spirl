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

# Download Office Dataset
gdown https://drive.google.com/uc?id=1FOE1kiU71nB-3KCDuxGqlAqRQbKmSk80

# Download Maze Dataset in smaller chunks (~3GB each)
gdown https://drive.google.com/uc?id=15fjt8QMC6xMpqTMftLF1RCWAe3jvklGb
gdown https://drive.google.com/uc?id=1aTz94EJYPU5A-h-EV5CxCIfd8io2e-EH
gdown https://drive.google.com/uc?id=1EiahoGgGiS7ol-Xx-DHIiEroZSeG9jEr
gdown https://drive.google.com/uc?id=1y5VafZN_95tHStEHKsoQvLrvw2y0pmRQ
``` 

Finally, unzip the downloaded zip files, update the `data_dir` in your config file to the unzipped directory with the dataset you want to use and then you are ready to start training! For the chunked maze data, make sure to copy all unzipped contents into a single folder called `maze`.

## Re-Generating Datasets

### Maze Dataset
To regenerate the maze dataset, [our fork of the D4RL repo](https://github.com/kpertsch/d4rl) needs to be cloned and installed.
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


### Office Dataset
To regenerate the office dataset, you need to first install [our fork of the Roboverse repo](https://github.com/VentusYue/roboverse).

To start generation, run:
```
python3 scripts/scripted_collect_parallel.py -p 10 -n 50000 -t 250 -e Widow250OfficeRand-v0 -pl tableclean -a table_clean -d office_TA
```
This will start 10 parallel data collection workers that will jointly collect 50k sequences into the directory `${DATA_DIR}/roboverse/office_TA`.
