import numpy as np
import contextlib
from collections import deque

from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.rl.utils.reward_fcns import sparse_threshold


class Sampler:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)

        self._env = env
        self._agent = agent
        self._logger = logger
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step, self._episode_reward = 0, 0

    def _default_hparams(self):
        return ParamDict({})

    def init(self, is_train):
        """Starts a new rollout. Render indicates whether output should contain image."""
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                self._episode_reset()

    def sample_action(self, obs):
        return self._agent.act(obs)

    def sample_batch(self, batch_size, is_train=True, global_step=None):
        """Samples an experience batch of the required size."""
        experience_batch = []
        step = 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while step < batch_size:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        if agent_output.action is None:
                            self._episode_reset(global_step)
                            continue
                        agent_output = self._postprocess_agent_output(agent_output)
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs = self._postprocess_obs(obs)
                        experience_batch.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                        ))

                        # update stored observation
                        self._obs = obs
                        step += 1; self._episode_step += 1; self._episode_reward += reward

                        # reset if episode ends
                        if done or self._episode_step >= self._max_episode_len:
                            if not done:    # force done to be True for timeout
                                experience_batch[-1].done = True
                            self._episode_reset(global_step)

        return listdict2dictlist(experience_batch), step

    def sample_episode(self, is_train, render=False):
        """Samples one episode from the environment."""
        self.init(is_train)
        episode, done = [], False
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while not done and self._episode_step < self._max_episode_len:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        if agent_output.action is None:
                            break
                        agent_output = self._postprocess_agent_output(agent_output)
                        if render:
                            render_obs = self._env.render()
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs = self._postprocess_obs(obs)
                        episode.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                            info=obj2np(info),
                        ))
                        if render:
                            episode[-1].update(AttrDict(image=render_obs))

                        # update stored observation
                        self._obs = obs
                        self._episode_step += 1
        episode[-1].done = True     # make sure episode is marked as done at final time step

        return listdict2dictlist(episode)

    def get_episode_info(self):
        episode_info = AttrDict(episode_reward=self._episode_reward,
                                episode_length=self._episode_step,)
        if hasattr(self._env, "get_episode_info"):
            episode_info.update(self._env.get_episode_info())
        return episode_info

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        if global_step is not None and self._logger is not None:    # logger is none in non-master threads
            self._logger.log_scalar_dict(self.get_episode_info(),
                                         prefix='train' if self._agent._is_train else 'val',
                                         step=global_step)
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._postprocess_obs(self._reset_env())
        self._agent.reset()

    def _reset_env(self):
        return self._env.reset()

    def _postprocess_obs(self, obs):
        """Optionally post-process observation."""
        return obs

    def _postprocess_agent_output(self, agent_output):
        """Optionally post-process / store agent output."""
        return agent_output


class HierarchicalSampler(Sampler):
    """Collects experience batches by rolling out a hierarchical agent. Aggregates low-level batches into HL batch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hl_obs, self.last_hl_action = None, None  # stores observation when last hl action was taken
        self.reward_since_last_hl = 0  # accumulates the reward since the last HL step for HL transition

    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        """Samples the required number of high-level transitions. Number of LL transitions can be higher."""
        hl_experience_batch, ll_experience_batch = [], []

        env_steps, hl_step = 0, 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while hl_step < batch_size or len(ll_experience_batch) <= 1:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        agent_output = self._postprocess_agent_output(agent_output)
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs = self._postprocess_obs(obs)

                        # update last step's 'observation_next' with HL action
                        if store_ll:
                            if ll_experience_batch:
                                ll_experience_batch[-1].observation_next = \
                                    self._agent.make_ll_obs(ll_experience_batch[-1].observation_next, agent_output.hl_action)

                            # store current step in ll_experience_batch
                            ll_experience_batch.append(AttrDict(
                                observation=self._agent.make_ll_obs(self._obs, agent_output.hl_action),
                                reward=reward,
                                done=done,
                                action=agent_output.action,
                                observation_next=obs,       # this will get updated in the next step
                            ))

                        # store HL experience batch if this was HL action or episode is done
                        if agent_output.is_hl_step or (done or self._episode_step >= self._max_episode_len-1):
                            if self.last_hl_obs is not None and self.last_hl_action is not None:
                                hl_experience_batch.append(AttrDict(
                                    observation=self.last_hl_obs,
                                    reward=self.reward_since_last_hl,
                                    done=done,
                                    action=self.last_hl_action,
                                    observation_next=obs,
                                ))
                                hl_step += 1
                                if done:
                                    hl_experience_batch[-1].reward += reward  # add terminal reward
                                if hl_step % 1000 == 0:
                                    print("Sample step {}".format(hl_step))
                            self.last_hl_obs = self._obs if self._episode_step == 0 else obs
                            self.last_hl_action = agent_output.hl_action
                            self.reward_since_last_hl = 0

                        # update stored observation
                        self._obs = obs
                        env_steps += 1; self._episode_step += 1; self._episode_reward += reward
                        self.reward_since_last_hl += reward

                        # reset if episode ends
                        if done or self._episode_step >= self._max_episode_len:
                            if not done:    # force done to be True for timeout
                                ll_experience_batch[-1].done = True
                                if hl_experience_batch:   # can potentially be empty 
                                    hl_experience_batch[-1].done = True
                            self._episode_reset(global_step)


        return AttrDict(
            hl_batch=listdict2dictlist(hl_experience_batch),
            ll_batch=listdict2dictlist(ll_experience_batch[:-1]),   # last element does not have updated obs_next!
        ), env_steps

    def _episode_reset(self, global_step=None):
        super()._episode_reset(global_step)
        self.last_hl_obs, self.last_hl_action = None, None
        self.reward_since_last_hl = 0


class ImageAugmentedSampler(Sampler):
    """Appends image rendering to raw observation."""
    def _postprocess_obs(self, obs):
        img = self._env.render().transpose(2, 0, 1) * 2. - 1.0
        return np.concatenate((obs, img.flatten()))


class MultiImageAugmentedSampler(Sampler):
    """Appends multiple past images to current observation."""
    def _episode_reset(self, global_step=None):
        self._past_frames = deque(maxlen=self._hp.n_frames)     # build ring-buffer of past images
        super()._episode_reset(global_step)

    def _postprocess_obs(self, obs):
        img = self._env.render().transpose(2, 0, 1) * 2. - 1.0
        if not self._past_frames:   # initialize past frames with N copies of current frame
            [self._past_frames.append(img) for _ in range(self._hp.n_frames - 1)]
        self._past_frames.append(img)
        stacked_img = np.concatenate(list(self._past_frames), axis=0)
        return np.concatenate((obs, stacked_img.flatten()))


class ACImageAugmentedSampler(ImageAugmentedSampler):
    """Adds no-op renders to make sure agent-centric camera reaches agent."""
    def _reset_env(self):
        obs = super()._reset_env()
        for _ in range(100):  # so that camera can "reach" agent
            self._env.render(mode='rgb_array')
        return obs


class ACMultiImageAugmentedSampler(MultiImageAugmentedSampler, ACImageAugmentedSampler):
    def _reset_env(self):
        return ACImageAugmentedSampler._reset_env(self)


class ImageAugmentedHierarchicalSampler(HierarchicalSampler, ImageAugmentedSampler):
    def _postprocess_obs(self, *args, **kwargs):
        return ImageAugmentedSampler._postprocess_obs(self, *args, **kwargs)


class MultiImageAugmentedHierarchicalSampler(HierarchicalSampler, MultiImageAugmentedSampler):
    def _postprocess_obs(self, *args, **kwargs):
        return MultiImageAugmentedSampler._postprocess_obs(self, *args, **kwargs)

    def _episode_reset(self, *args, **kwargs):
        return MultiImageAugmentedSampler._episode_reset(self, *args, **kwargs)


class ACImageAugmentedHierarchicalSampler(ImageAugmentedHierarchicalSampler, ACImageAugmentedSampler):
    def _reset_env(self):
        return ACImageAugmentedSampler._reset_env(self)


class ACMultiImageAugmentedHierarchicalSampler(MultiImageAugmentedHierarchicalSampler,
                                               ACImageAugmentedHierarchicalSampler):
    def _reset_env(self):
        return ACImageAugmentedHierarchicalSampler._reset_env(self)
