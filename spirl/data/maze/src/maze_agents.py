import matplotlib.pyplot as plt
import numpy as np

from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent, ACSkillSpaceAgent


class MazeAgent:
    """Adds replay logging function."""
    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        if step > self.replay_buffer.capacity:
            return   # visualization does not work if earlier samples were overridden

        # get data
        size = self.replay_buffer.size
        states = self.replay_buffer.get().observation[:size, :2]

        fig = plt.figure()
        plt.scatter(states[:, 0], states[:, 1], s=5, c=np.arange(size), cmap='Blues')
        plt.axis("equal")
        logger.log_plot(fig, "replay_vis", step)
        plt.close(fig)


class MazeSkillSpaceAgent(SkillSpaceAgent, MazeAgent):
    """Collects samples in replay buffer for visualization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = self._hp.replay(self._hp.replay_params)

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer (used during warmup)."""
        self.replay_buffer.append(experience_batch)
        return SkillSpaceAgent.add_experience(self, experience_batch)

    def update(self, experience_batch):
        self.replay_buffer.append(experience_batch)
        return SkillSpaceAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        MazeAgent.visualize(self, logger, rollout_storage, step)
        SkillSpaceAgent.visualize(self, logger, rollout_storage, step)


class MazeACSkillSpaceAgent(MazeSkillSpaceAgent, ACSkillSpaceAgent):
    """Maze version of ACSkillSpaceAgent for obs with agent-centric prior input."""
    def _act(self, obs):
        return ACSkillSpaceAgent._act(self, obs)


class MazeSACAgent(SACAgent, MazeAgent):
    def visualize(self, logger, rollout_storage, step):
        MazeAgent.visualize(self, logger, rollout_storage, step)
        SACAgent.visualize(self, logger, rollout_storage, step)


class MazeActionPriorSACAgent(ActionPriorSACAgent, MazeAgent):
    def visualize(self, logger, rollout_storage, step):
        MazeAgent.visualize(self, logger, rollout_storage, step)
        ActionPriorSACAgent.visualize(self, logger, rollout_storage, step)


class MazeNoUpdateAgent(MazeAgent, SACAgent):
    """Only logs rollouts, does not update policy."""
    def update(self, experience_batch):
        self.replay_buffer.append(experience_batch)
        return {}


class MazeACActionPriorSACAgent(ActionPriorSACAgent, MazeAgent):
    def __init__(self, *args, **kwargs):
        ActionPriorSACAgent.__init__(self, *args, **kwargs)
        from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer
        # TODO: don't hardcode this for res 32x32
        self.vis_replay_buffer = SplitObsUniformReplayBuffer({'capacity': 1e7, 'unused_obs_size': 6144,})

    def update(self, experience_batch):
        self.vis_replay_buffer.append(experience_batch)
        return ActionPriorSACAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)
        ActionPriorSACAgent.visualize(self, logger, rollout_storage, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        if step > self.replay_buffer.capacity:
            return   # visualization does not work if earlier samples were overridden

        # get data
        size = self.vis_replay_buffer.size
        states = self.vis_replay_buffer.get().observation[:size, :2]

        fig = plt.figure()
        plt.scatter(states[:, 0], states[:, 1], s=5, c=np.arange(size), cmap='Blues')
        plt.axis("equal")
        logger.log_plot(fig, "replay_vis", step)
        plt.close(fig)
