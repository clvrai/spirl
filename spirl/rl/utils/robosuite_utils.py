from spirl.data.block_stacking.src.robosuite.environments.base import MujocoEnv
from mujoco_py import MjSim, MjRenderContextOffscreen
from spirl.data.block_stacking.src.robosuite.utils import MujocoPyRenderer


class FastResetMujocoEnv(MujocoEnv):
    """Only loads the mujoco XML file once to allow for quicker resets."""
    def _reset_internal(self):
        """Resets simulation internal configurations."""
        # instantiate simulation from MJCF model
        self._load_model()
        if not hasattr(self, "mjpy_model"):
            self.mjpy_model = self.model.get_model(mode="mujoco_py")
            self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                1 if self.render_visual_mesh else 0
            )

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False
