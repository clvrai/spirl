import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from spirl.utils.vis_utils import plot_graph


class Logger:
    def __init__(self, log_dir, n_logged_samples=3, summary_writer=None):
        self._log_dir = log_dir
        self._n_logged_samples = n_logged_samples
        if summary_writer is not None:
            self._summ_writer = summary_writer
        else:
            self._summ_writer = SummaryWriter(log_dir)

    def log_scalar(self, scalar, name, step, phase):
        self._summ_writer.add_scalar('{}_{}'.format(name, phase), scalar, step)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_images(self, image, name, step, phase):
        image = self._format_input(image)
        self._check_size(image, 4)   # [N, C, H, W]
        self._loop_batch(self._summ_writer.add_image, '{}_{}'.format(name, phase), image, step)

    def log_gif(self, gif_frames, name, step, phase):
        if isinstance(gif_frames, list): gif_frames = np.concatenate(gif_frames)
        gif_frames = self._format_input(gif_frames)
        assert len(gif_frames.shape) == 4, "Need [T, C, H, W] input tensor for single video logging!"
        gif_frames = gif_frames.unsqueeze(0)    # add an extra dimension to get grid of size 1
        self._summ_writer.add_video('{}_{}'.format(name, phase), gif_frames, step, fps=10)
        
    def log_graph(self, array, name, step, phase):
        """array gets plotted with plt.plot"""
        im = torch.tensor(plot_graph(array).transpose(2, 0, 1))
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def _loop_batch(self, fn, name, val, *argv, **kwargs):
        """Loops the logging function n times."""
        for log_idx in range(min(self._n_logged_samples, len(val))):
            name_i = os.path.join(name, "_%d" % log_idx)
            fn(name_i, val[log_idx], *argv, **kwargs)

    def visualize(self, *args, **kwargs):
        """Subclasses can implement this method to visualize training results."""
        pass

    @staticmethod
    def _check_size(val, size):
        if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert len(val.shape) == size, "Size of tensor does not fit required size, {} vs {}".format(len(val.shape),
                                                                                                        size)
        elif isinstance(val, list):
            assert len(val[0].shape) == size - 1, "Size of list element does not fit required size, {} vs {}".format(
                len(val[0].shape), size - 1)
        else:
            raise NotImplementedError("Input type {} not supported for dimensionality check!".format(type(val)))
        if (val[0].shape[1] > 10000) or (val[0].shape[2] > 10000):
            print("Logging very large image with size {}px.".format(max(val[0].shape[1], val[0].shape[2])))
            raise ValueError("This might be a bit too much")

    @staticmethod
    def _format_input(arr):
        if not isinstance(arr, torch.Tensor): arr = torch.tensor(arr)
        if not (arr.shape[1] == 3 or arr.shape[1] == 1): arr = arr.permute(0, 3, 1, 2)
        arr = arr.float()
        return arr

    def __del__(self):
        self._summ_writer.close()
        print("Closed summary writer.")


if __name__ == "__main__":
    logger = Logger(log_dir="./summaries")
    for step in range(10):
        print("Running step %d" % step)
        dummy_data = torch.rand([32, 10, 3, 64, 64])
        logger.log_scalar(dummy_data[0, 0, 0, 0, 0], name="scalar", step=step, phase="train")
        logger.log_scalars({
            'test1': dummy_data[0, 0, 0, 0, 0],
            'test2': dummy_data[0, 0, 0, 0, 1],
            'test3': dummy_data[0, 0, 0, 0, 2]
        }, group_name="scalar_group", step=step, phase="train")
        logger.log_images(dummy_data[:, 0], name="image", step=step, phase="train")
        logger.log_gif(dummy_data, name="video", step=step, phase="train")
        logger.log_graph(np.asarray([i for i in range(10)]), name="figure", step=step, phase="train")
    logger.dump_scalars()
    print("Done!")


