import d4rl
import gym
import numpy as np

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict
from spirl.utils.video_utils import resize_video


class MazeStateSequenceDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = self.phase == "train"

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        self.n_samples = self.dataset['infos/qpos'].shape[0]
        self.subseq_len = self.spec.subseq_len

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_samples)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_samples)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_samples)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_samples)
            self.end = self.n_samples

    def __getitem__(self, index):
        # sample start index in data range
        start_idx = self._sample_start_idx()
        output = AttrDict(
            states=np.concatenate([self.dataset["infos/qpos"][start_idx : start_idx + self.subseq_len],
                                   self.dataset["infos/qvel"][start_idx: start_idx + self.subseq_len]], axis=-1),
            actions=self.dataset['actions'][start_idx : start_idx + self.subseq_len - 1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        return output

    def _sample_start_idx(self):
        return np.random.randint(self.start, self.end - self.subseq_len - 1)

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.n_samples / self.subseq_len)


class AgentCentricRenderMazeSequenceDataset(MazeStateSequenceDataset):
    """Renders agent-centric top-down view on-the fly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._render_env = gym.make(self.spec.env_name[:-3] + '-agent_centric' + self.spec.env_name[-3:])

    def __getitem__(self, item):
        output = super().__getitem__(item)
        output.images = self._render_action_seq_agent_centric(output.states).transpose(0, 3, 1, 2)
        return output

    def _render_action_seq_agent_centric(self, pos_seq):
        # initialize env to correct start state
        self._render_env.reset()
        self._render_env.set_state(pos_seq[0, :2], pos_seq[0, 2:])
        [self._render_env.render(mode='rgb_array') for _ in range(100)]     # so that camera can "reach" agent

        # render rollout
        imgs = [self._render_env.render(mode='rgb_array')]
        for pos in pos_seq[1:]:
            self._render_env.set_state(pos[:2], pos[2:])
            imgs.append(self._render_env.render(mode='rgb_array'))
        imgs = resize_video(np.stack(imgs), (self.spec.res, self.spec.res)) / 255. * 2. - 1.0
        return np.asarray(imgs, dtype=np.float32)

    def __len__(self):
        return int(super().__len__() / 10)
