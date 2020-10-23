import os
from contextlib import contextmanager
import numpy as np
import torch
from skimage.io import imsave

from spirl.utils.eval_utils import ssim, psnr, mse
from spirl.utils.general_utils import batchwise_index, AttrDict, timed, dict_concat
from spirl.utils.vis_utils import make_gif_strip


class TopOfNEvaluator:
    """Implements a basic evaluator."""
    N_PLOTTED_ELEMENTS = 5
    LOWER_IS_BETTER_METRICS = ['mse']
    HIGHER_IS_BETTER_METRICS = ['psnr', 'ssim']

    def __init__(self, hp, logdir, top_of_n, top_comp_metric, tb_logger=None):
        self._hp = hp
        self._logdir = logdir + '/eval'
        self._logger = FileEvalLogger(self._logdir) if tb_logger is None else TBEvalLogger(logdir, tb_logger)
        self._top_of_n = top_of_n
        self._top_comp_metric = top_comp_metric
        self.batch_eval_buffer = None      # holds evaluation results for the current batch
        self.full_eval_buffer = None        # holds evaluation results across batches

    def reset(self):
        self.batch_eval_buffer = None
        self.full_eval_buffer = None

    def _erase_eval_buffer(self):
        def get_init_array(val):
            return val * np.ones((self._hp.batch_size, self._top_of_n))
        self.batch_eval_buffer = AttrDict()
        for metric_name in self.metrics:
            default_value = 0. if metric_name in TopOfNEvaluator.HIGHER_IS_BETTER_METRICS else np.inf
            self.batch_eval_buffer[metric_name] = get_init_array(default_value)
        self.batch_eval_buffer.aux_outputs = np.empty(self._hp.batch_size, dtype=np.object)

    def eval_single(self, inputs, model_output, sample_idx):
        bsize = self._hp.batch_size
        for b in range(bsize):
            for metric_name, metric_fcn in self.metrics.items():
                self.batch_eval_buffer[metric_name][b, sample_idx] = metric_fcn(inputs, model_output, b)

            if self._top_of_n == 1 or \
                self._is_better(self.batch_eval_buffer[self._top_comp_metric][b, sample_idx],
                                self.batch_eval_buffer[self._top_comp_metric][b], self._top_comp_metric):
                self.batch_eval_buffer.aux_outputs[b] = self._store_aux_outputs(inputs, model_output, b)

    @timed("Eval time for batch: ")
    def eval(self, inputs, model):
        self._erase_eval_buffer()
        for n in range(self._top_of_n):
            model_output = model(inputs)
            self.eval_single(inputs, model_output, sample_idx=n)
        self._flush_eval_buffer()

    def _flush_eval_buffer(self):
        if self.full_eval_buffer is None:
            self.full_eval_buffer = self.batch_eval_buffer
        else:
            dict_concat(self.full_eval_buffer, self.batch_eval_buffer)

    def dump_results(self, it):
        self.dump_metrics(it)
        self.dump_outputs(it)
        self.reset()

    def dump_metrics(self, it):
        with self._logger.log_to('results', it, 'metric'):
            best_idxs = 0 if self._top_of_n == 1 else self._get_best_idxs(self.full_eval_buffer[self._top_comp_metric])
            print_st = []
            for metric in sorted(self.metrics):
                vals = self.full_eval_buffer[metric]
                best_vals = batchwise_index(vals, best_idxs)
                print_st.extend(['{}: {}'.format(metric, best_vals.mean())])
                self._logger.log(metric, vals if self._top_of_n > 1 else None, best_vals)
            print(*print_st, sep=', ')

    def _is_better(self, val, other, metric):
        """Comparison function for different metrics.
           returns True if val is "better" than any of the values in the array other
        """
        if metric in self.LOWER_IS_BETTER_METRICS:
            return np.all(val <= other)
        elif metric in self.HIGHER_IS_BETTER_METRICS:
            return np.all(val >= other)
        else:
            raise ValueError("Currently only support comparison on the following metrics: {}. Got {}."
                             .format(self.LOWER_IS_BETTER_METRICS + self.HIGHER_IS_BETTER_METRICS, metric))

    def _get_best_idxs(self, vals):
        assert len(vals.shape) == 2     # assumes batch in first dimension, N samples in second dim
        if self._top_comp_metric in self.LOWER_IS_BETTER_METRICS:
            return np.argmin(vals, axis=1)
        else:
            return np.argmax(vals, axis=1)

    @property
    def metrics(self):
        """Defines a dict of metric names and the associated function handles. Each metric function needs to follow
        the definition: fnc(inputs, model_outputs, batch_idx)."""
        raise NotImplementedError("This needs to be implemented by the inheriting class.")

    def _store_aux_outputs(self, inputs, model_outputs, batch_idx):
        """Option to store model outputs of the best sample.
        Batch index indicates which output is currently being processed."""
        return AttrDict()

    def dump_outputs(self, it):
        """Can be used to visualize / log any additional outputs like produced eval samples etc."""
        pass


class Evaluator(TopOfNEvaluator):
    """Evaluator class with a single sample per validation sequence."""
    def __init__(self, hp, logdir, top_of_n, top_comp_metric, *args, **kwargs):
        if top_of_n > 1 or top_comp_metric is not None:
            raise ValueError("Cannot instantiate TopOf1 evaluator when params are set this way!")
        super().__init__(hp, logdir, top_of_n=1, top_comp_metric=None, *args, **kwargs)


class ImageEvaluator(Evaluator):
    """Implements Evaluator for image data that evaluates MSE, PSNR, SSIM."""
    @property
    def metrics(self):
        return AttrDict(mse=self._mse,
                        psnr=self._psnr,
                        ssim=self._ssim,)

    def _mse(self, inputs, model_outputs, batch_idx):
        return mse(model_outputs.output_imgs[batch_idx], inputs.images[batch_idx, -model_outputs.output_imgs.shape[1]:])

    def _psnr(self, inputs, model_outputs, batch_idx):
        return psnr(model_outputs.output_imgs[batch_idx], inputs.images[batch_idx, -model_outputs.output_imgs.shape[1]:])

    def _ssim(self, inputs, model_outputs, batch_idx):
        return ssim(model_outputs.output_imgs[batch_idx], inputs.images[batch_idx, -model_outputs.output_imgs.shape[1]:])

    def _store_aux_outputs(self, inputs, model_outputs, batch_idx):
        return AttrDict(gt=inputs.images[batch_idx], estimate=model_outputs.output_imgs[batch_idx])

    def dump_outputs(self, it):
        """Here we could implement any image logging."""
        pass


class TopOfNSequenceEvaluator(TopOfNEvaluator):
    """Implements Evaluator for (non-image) sequence data that evaluates MSE."""
    @property
    def metrics(self):
        return AttrDict(mse=self._mse,)

    def _mse(self, inputs, model_outputs, batch_idx):
        return mse(model_outputs.reconstruction[batch_idx], inputs.observations[batch_idx, -model_outputs.reconstruction.shape[1]:])

    def _store_aux_outputs(self, inputs, model_outputs, batch_idx):
        return AttrDict(gt=inputs.observations[batch_idx], estimate=model_outputs.reconstruction[batch_idx])

    def dump_outputs(self, it):
        """Here we could implement any image logging."""
        pass


class SequenceEvaluator(TopOfNSequenceEvaluator):
    """Evaluator class with a single sample per validation sequence."""
    def __init__(self, hp, logdir, top_of_n, top_comp_metric, *args, **kwargs):
        if top_of_n > 1:
            raise ValueError("Cannot instantiate TopOf1 evaluator when params are set this way!")
        super().__init__(hp, logdir, top_of_n=1, top_comp_metric=top_comp_metric, *args, **kwargs)



class VideoEvaluator(ImageEvaluator):
    def dump_outputs(self, it):
        """Only the output saving function will likely need to change."""
        pass


class DummyEvaluator(Evaluator):
    def eval(self, inputs, model):
        return model(inputs)

    def dump_results(self, it):
        pass


class EvalLogger:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self.log_target = None
        self.log_type = None
        self.log_tag = None
        self.log_counter = None

    @contextmanager
    def log_to(self, tag, it, type):
        """Sets logging context (e.g. what file to log to)."""
        raise NotImplementedError

    def log(self, *vals):
        """Implements logging within the 'log_to' context."""
        assert self.log_target is not None      # cannot log without 'log_to' context
        if self.log_type == 'metric':
            self._log_metric(*vals)
        elif self.log_type == 'image':
            self._log_img(*vals)
        elif self.log_type == 'array':
            self._log_array(*vals)
        elif self.log_type == 'gif':
            self._log_gif(*vals)
        elif self.log_type == 'graph':
            self._log_graph(*vals)
        self.log_counter += 1

    def _log_metric(self, name, vals, best_vals):
        raise NotImplementedError

    def _log_img(self, img):
        raise NotImplementedError

    def _log_array(self, array):
        np.save(os.path.join(self.log_target, "{}_{}.npy".format(self.log_tag, self.log_counter)), array)

    def _log_gif(self, gif):
        pass

    def _log_graph(self, array):
        raise NotImplementedError

    def _make_dump_dir(self, tag, it):
        dump_dir = os.path.join(self._log_dir, '{}/it_{}'.format(tag, it))
        if not os.path.exists(dump_dir): os.makedirs(dump_dir)
        return dump_dir


class FileEvalLogger(EvalLogger):
    """Logs evaluation results on disk."""
    @contextmanager
    def log_to(self, tag, it, log_type):
        """Creates logging file."""
        self.log_type, self.log_tag, self.log_counter = log_type, tag, 0
        if log_type == 'metric':
            self.log_target = open(os.path.join(self._log_dir, '{}_{}.txt'.format(tag, it)), 'w')
        elif log_type == 'image' or log_type == 'array':
            self.log_target = self._make_dump_dir(tag, it)
        elif log_type in ('gif', 'graph'):
            self.log_target = 'no log'
        else:
            raise ValueError("Type {} is not supported for logging in eval!".format(log_type))
        yield
        if log_type == 'metric':
            self.log_target.close()
        self.log_target, self.log_type, self.log_tag, self.log_counter = None, None, None, None

    def _log_metric(self, name, vals, best_vals):
        str = 'mean {} {}, standard error of the mean (SEM) {}'.format(name, best_vals.mean(), best_vals.std())
        str += ', mean std of 100 samples {}\n'.format(vals.std(axis=1).mean()) if vals is not None else '\n'
        self.log_target.write(str)
        print(str)

    def _log_img(self, img):
        assert img.max() < 1.0  # expect image to be in range [-1...1]
        imsave(os.path.join(self.log_target, "{}_{}.png".format(self.log_tag, self.log_counter)), (img + 1) / 2)


class TBEvalLogger(EvalLogger):
    """Logs evaluation results to Tensorboard."""
    def __init__(self, log_dir, tb_logger):
        super().__init__(log_dir)
        self._tb_logger = tb_logger
        self.log_step = None

    @contextmanager
    def log_to(self, tag, it, log_type):
        self.log_type, self.log_tag, self.log_counter, self.log_step = log_type, tag, 0, it
        if log_type == 'array':
            self.log_target = self._make_dump_dir(tag, it)
        else:
            self.log_target = 'TB'
        yield
        self.log_target, self.log_type, self.log_tag, self.log_counter, self.log_step = None, None, None, None, None

    def _log_metric(self, name, vals, best_vals):
        self._tb_logger.log_scalar(best_vals.mean(), self.group_tag + '/metric/{}/top100_mean'.format(name), self.log_step, '')
        self._tb_logger.log_scalar(best_vals.std(), self.group_tag + '/verbose/{}/top100_std'.format(name), self.log_step, '')
        if vals is not None:
            self._tb_logger.log_scalar(vals.mean(), self.group_tag + '/verbose/{}/all100_mean'.format(name), self.log_step, '')
            self._tb_logger.log_scalar(vals.std(axis=1).mean(), self.group_tag + '/verbose/{}/all100_std'.format(name), self.log_step, '')

    def _log_img(self, img):
        assert img.max() <= 1.0 and img.shape[-1] in (1, 3)  # expect img in range [-1...1], [H, W, C]
        if not isinstance(img, torch.Tensor): img = torch.tensor(img)
        img = (img.permute(2, 0, 1) + 1) / 2
        self._tb_logger.log_images(img[None], self.group_tag + '/{}'.format(self.log_counter), self.log_step, '')

    def _log_gif(self, gif):
        self._tb_logger.log_gif(gif, self.group_tag + '/{}'.format(self.log_counter), self.log_step, '')

    def _log_graph(self, array):
        self._tb_logger.log_graph(array, self.group_tag + '/{}'.format(self.log_counter), self.log_step, '')

    @property
    def group_tag(self):
        assert self.log_tag is not None     # need to set logging context first
        return 'eval/{}'.format(self.log_tag)
