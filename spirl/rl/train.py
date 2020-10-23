import torch
import os
import imp
import json
from tqdm import tqdm
from collections import defaultdict

from spirl.rl.components.params import get_args
from spirl.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from spirl.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum
from spirl.rl.utils.wandb import WandBLogger
from spirl.rl.utils.rollout_utils import RolloutSaver
from spirl.rl.components.sampler import Sampler
from spirl.rl.components.replay_buffer import RolloutStorage

WANDB_PROJECT_NAME = 'comp_imitation'
WANDB_ENTITY_NAME = 'clvr'


class RLTrainer:
    """Sets up RL training loop, instantiates all components, runs training."""
    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        update_with_mpi_config(self.conf)   # self.conf.mpi = AttrDict(is_chef=True)
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed   # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        # set up logging
        if self.is_chef:
            print("Running base worker.")
            self.logger = self.setup_logging(self.conf, self.log_dir)
        else:
            print("Running worker {}, disabled logging.".format(self.conf.mpi.rank))
            self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        if 'task_params' in self.conf.env: self.conf.env.task_params.seed=self._hp.seed
        if 'general' in self.conf: self.conf.general.seed=self._hp.seed
        self.env = self._hp.environment(self.conf.env)
        self.conf.agent.env_params = self.env.agent_params      # (optional) set params from env for agent
        if self.is_chef:
            pretty_print(self.conf)

        # build agent (that holds actor, critic, exposes update method)
        self.conf.agent.num_workers = self.conf.mpi.num_workers
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 0     # no warmup if we reload from checkpoint!

        # start training/evaluation
        if args.mode == 'train':
            self.train(start_epoch)
        elif args.mode == 'val':
            self.val()
        else:
            self.generate_rollouts()

    def _default_hparams(self):
        default_dict = ParamDict({
            'seed': None,
            'agent': None,
            'data_dir': None,  # directory where dataset is in
            'environment': None,
            'sampler': Sampler,     # sampler type used
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'max_rollout_len': 1000,  # maximum length of the performed rollout
            'n_steps_per_update': 1,     # number of env steps collected per policy update
            'n_steps_per_epoch': 20000,       # number of env steps per epoch
            'log_output_per_epoch': 100,  # log the non-image/video outputs N times per epoch
            'log_images_per_epoch': 4,    # log images/videos N times per epoch
            'logging_target': 'wandb',    # where to log results to
            'n_warmup_steps': 0,    # steps of warmup experience collection before training
        })
        return default_dict

    def train(self, start_epoch):
        """Run outer training loop."""
        if self._hp.n_warmup_steps > 0:
            self.warmup()

        for epoch in range(start_epoch, self._hp.num_epochs):
            print("Epoch {}".format(epoch))
            self.train_epoch(epoch)

            if not self.args.dont_save and self.is_chef:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                }, os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))
                self.agent.save_state(self._hp.exp_path)
                self.val()

    def train_epoch(self, epoch):
        """Run inner training loop."""
        # sync network parameters across workers
        if self.conf.mpi.num_workers > 1:
            self.agent.sync_networks()

        # initialize timing
        timers = defaultdict(lambda: AverageTimer())

        self.sampler.init(is_train=True)
        ep_start_step = self.global_step
        while self.global_step - ep_start_step < self._hp.n_steps_per_epoch:
            with timers['batch'].time():
                # collect experience
                with timers['rollout'].time():
                    experience_batch, env_steps = self.sampler.sample_batch(batch_size=self._hp.n_steps_per_update, global_step=self.global_step)
                    self.global_step += mpi_sum(env_steps)

                # update policy
                with timers['update'].time():
                    agent_outputs = self.agent.update(experience_batch)
                    self.n_update_steps += 1

                # log results
                with timers['log'].time():
                    if self.is_chef and self.log_outputs_now:
                        self.agent.log_outputs(agent_outputs, None, self.logger,
                                               log_images=False, step=self.global_step)
                        self.print_train_update(epoch, agent_outputs, timers)

    def val(self):
        """Evaluate agent."""
        val_rollout_storage = RolloutStorage()
        with self.agent.val_mode():
            with torch.no_grad():
                with timing("Eval rollout time: "):
                    for _ in range(WandBLogger.N_LOGGED_SAMPLES):   # for efficiency instead of self.args.n_val_samples
                        val_rollout_storage.append(self.sampler.sample_episode(is_train=False, render=True))

        rollout_stats = val_rollout_storage.rollout_stats()
        if self.is_chef:
            with timing("Eval log time: "):
                self.agent.log_outputs(rollout_stats, val_rollout_storage,
                                       self.logger, log_images=True, step=self.global_step)
            print("Evaluation Avg_Reward: {}".format(rollout_stats.avg_reward))
        del val_rollout_storage

    def generate_rollouts(self):
        """Generate rollouts and save to hdf5 files."""
        print("Saving {} rollouts to directory {}...".format(self.args.n_val_samples, self.args.save_dir))
        saver = RolloutSaver(self.args.save_dir)
        n_success = 0
        n_total = 0
        with self.agent.val_mode():
            with torch.no_grad():
                for _ in tqdm(range(self.args.n_val_samples)):
                    while True:     # keep producing rollouts until we get a valid one
                        episode = self.sampler.sample_episode(is_train=False, render=True)
                        valid = not hasattr(self.agent, 'rollout_valid') or self.agent.rollout_valid
                        n_total += 1
                        if valid:
                            n_success += 1
                            break
                    saver.save_rollout_to_file(episode)
        print("Success rate: {:d} / {:d} = {:.3f}%".format(n_success, n_total, float(n_success) / n_total * 100))

    def warmup(self):
        """Performs pre-training warmup experience collection with random policy."""
        print(f"Warmup data collection for {self._hp.n_warmup_steps} steps...")
        with self.agent.rand_act_mode():
            self.sampler.init(is_train=True)
            warmup_experience_batch, _ = self.sampler.sample_batch(batch_size=self._hp.n_warmup_steps)
        self.agent.add_experience(warmup_experience_batch)
        print("...Warmup done!")

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and agent configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.agent = conf_module.agent_config
        conf.agent.device = self.device

        # data config
        conf.data = conf_module.data_config

        # environment config
        conf.env = conf_module.env_config
        conf.env.device = self.device       # add device to env config as it directly returns tensors

        # sampler config
        conf.sampler = conf_module.sampler_config if hasattr(conf_module, 'sampler_config') else AttrDict({})

        # model loading config
        conf.ckpt_path = conf.agent.checkpt_path if 'checkpt_path' in conf.agent else None

        # load notes if there are any
        if self.args.notes != '':
            conf.notes = self.args.notes
        else:
            try:
                conf.notes = conf_module.notes
            except:
                conf.notes = ''

        # load config overwrites
        if self.args.config_override != '':
            for override in self.args.config_override.split(','):
                key_str, value_str = override.split('=')
                keys = key_str.split('.')
                curr = conf
                for key in keys[:-1]:
                    curr = curr[key]
                curr[keys[-1]] = type(curr[keys[-1]])(value_str)

        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))

            # setup logger
            logger = None
            if self.args.mode == 'train':
                exp_name = f"{os.path.basename(self.args.path)}_{self.args.prefix}" if self.args.prefix \
                    else os.path.basename(self.args.path)
                if self._hp.logging_target == 'wandb':
                    logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                         path=self._hp.exp_path, conf=conf)
                else:
                    raise NotImplementedError   # TODO implement alternative logging (e.g. TB)
            return logger

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        # TODO(karl): check whether that actually loads the optimizer too
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.agent,
                                           load_step=True, strict=self.args.strict_weight_loading)
        self.agent.load_state(self._hp.exp_path)
        self.agent.to(self.device)
        return start_epoch

    def print_train_update(self, epoch, agent_outputs, timers):
        print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                  self._hp.exp_path))
        print('Train Epoch: {} [It {}/{} ({:.0f}%)]'.format(
            epoch, self.global_step, self._hp.n_steps_per_epoch * self._hp.num_epochs,
                                     100. * self.global_step / (self._hp.n_steps_per_epoch * self._hp.num_epochs)))
        print('avg time for rollout: {:.2f}s, update: {:.2f}s, logs: {:.2f}s, total: {:.2f}s'
              .format(timers['rollout'].avg, timers['update'].avg, timers['log'].avg,
                      timers['rollout'].avg + timers['update'].avg + timers['log'].avg))
        togo_train_time = timers['batch'].avg * (self._hp.num_epochs * self._hp.n_steps_per_epoch - self.global_step) \
                          / self._hp.n_steps_per_update / 3600.
        print('ETA: {:.2f}h'.format(togo_train_time))

    @property
    def log_outputs_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                       / self._hp.log_output_per_epoch) == 0 \
                    or self.log_images_now

    @property
    def log_images_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                       / self._hp.log_images_per_epoch) == 0

    @property
    def is_chef(self):
        return self.conf.mpi.is_chef


if __name__ == '__main__':
    RLTrainer(args=get_args())


