import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from spirl.utils.pytorch_utils import pad_seq, stack_with_separator


def fig2img(fig):
    """Converts a given figure handle to a 3-channel numpy image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.


def plot_graph(array, h=400, w=400, dpi=10, linewidth=3.0):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    plt.xlim(0, array.shape[0] - 1)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.plot(array)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img


def tensor2np(tensor, n_logged_samples=None):
    if tensor is None: return tensor
    if n_logged_samples is None: return tensor.data.cpu().numpy()
    return tensor[:n_logged_samples].data.cpu().numpy()


def imgtensor2np(tensor, n_logged_samples=None):
    if tensor is None: return tensor
    return (tensor2np(tensor, n_logged_samples) + 1 ) / 2


def np2imgtensor(array, device, n_logged_samples=None):
    if array is None: return array
    if n_logged_samples is not None: array = array[:n_logged_samples]
    return torch.tensor(array * 2 - 1, device=device)


def action2img(action, res, channels, scale=50):
    """Plots 2D-actions into an arrow image.
       scale is the stretching factor."""
    assert action.size == 2   # can only plot 2-dimensional actions
    img = np.zeros((res, res, channels), dtype=np.float32).copy()
    start_pt = res /2 * np.ones((2,))
    end_pt = start_pt + action * scale * (res /2 - 1) * np.array([1, -1])     # swaps last dimension
    np2pt = lambda x: tuple(np.asarray(x, int))
    img = cv2.arrowedLine(img, np2pt(start_pt), np2pt(end_pt), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2)
    return img * 255.0


def batch_action2img(actions, res, channels):
    batch, seq_len, _ = actions.shape
    im = np.empty((batch, seq_len, res, res, channels), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            im[b, s] = action2img(actions[b, s], res, channels)
    return im


def make_image_strip(imgs, n_logged_samples=5):
    """Creates image strip with separators from list of images [each element of list makes one row]."""
    plot_imgs = stack_with_separator(imgs, dim=2)[:n_logged_samples]
    return stack_with_separator([t[0] for t in torch.split(plot_imgs, 1)], dim=2)


def make_image_seq_strip(imgs, n_logged_samples=5):
    """Creates image strip where each row contains full rollout of sequence [each element of list makes one row]."""
    plot_imgs = stack_with_separator(imgs, dim=3)[:n_logged_samples]
    return stack_with_separator([t[:, 0] for t in torch.split(plot_imgs, 1, dim=1)], dim=3)


def make_gif_strip(seqs, n_logged_samples=5):
    """Fuse sequences in list vertically + batch horizontally.
    :arg seqs: list of sequence tensors [batch, time, channel, height, width]
    :arg n_logged_samples: how many sequences should be logged horizontally in the strip
    """
    plot_imgs = stack_with_separator(seqs, dim=3)[:n_logged_samples]
    return stack_with_separator([t[0] for t in torch.split(plot_imgs, 1)], dim=3)


def make_padded_gif_strip(seqs, n_logged_samples=5, max_seq_len=None):
    """Same as 'make_gif_strip' but pads all sequences to max length.
    :arg max_seq_len: sequence length that all seqs are padded to, if None uses maximum of all sequences.
    """
    if max_seq_len is None:
        max_seq_len = max([seq.shape[1] for seq in seqs])
    seqs = [pad_seq(seq, length=max_seq_len) for seq in seqs]
    return make_gif_strip(seqs, n_logged_samples)


def int2color(int_array, n_max_ints=5):
    """Returns color-mapped version of integer array."""
    orig_shape = int_array.shape
    colors = plt.cm.jet(int_array.reshape(-1, 1) / n_max_ints)[..., :3]     # omit alpha channel
    output = np.asarray(colors.reshape(list(orig_shape) + [3]), dtype=np.float32)
    return output


def add_caption_to_img(img, info, name=None, flip_rgb=False):
    """ Adds caption to an image. info is dict with keys and text/array.
        :arg name: if given this will be printed as heading in the first line
        :arg flip_rgb: set to True for inputs with BGR color channels
    """
    offset = 12

    frame = img * 255.0 if img.max() <= 1.0 else img
    if flip_rgb:
        frame = frame[:, :, ::-1]

    # make frame larger if needed
    if frame.shape[0] < 300:
        frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)

    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((offset * (len(info.keys()) + 2), fwidth, 3))], 0)

    font_size = 0.4
    thickness = 1
    x, y = 5, fheight + 10
    if name is not None:
        cv2.putText(frame, '[{}]'.format(name),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 0), thickness, cv2.LINE_AA)
    for i, k in enumerate(info.keys()):
        v = info[k]
        key_text = '{}: '.format(k)
        (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, thickness)

        cv2.putText(frame, key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (66, 133, 244), thickness, cv2.LINE_AA)

        cv2.putText(frame, str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (100, 100, 100), thickness, cv2.LINE_AA)

    if flip_rgb:
        frame = frame[:, :, ::-1]

    return frame


def add_captions_to_seq(img_seq, info_seq, **kwargs):
    """Adds caption to sequence of image. info_seq is list of dicts with keys and text/array."""
    return [add_caption_to_img(img, info, name='Timestep {:03d}'.format(i), **kwargs) for i, (img, info) in enumerate(zip(img_seq, info_seq))]


def videos_to_grid(videos, num_per_row=5):
    """"Converts a numpy array of videos with shape (N, T, H, W, C) to a grid of videos."""
    assert len(videos.shape) == 5, "videos should have shape of length 5, but have shape {} instead".format(videos.shape)
    remainder = num_per_row - ((videos.shape[0] - 1) % num_per_row + 1)
    videos = np.concatenate([videos, np.zeros([remainder] + list(videos.shape[1:]))], axis=0)
    videos = videos.reshape([int(videos.shape[0]/num_per_row), num_per_row] + list(videos.shape[1:])).transpose((2,0,3,1,4,5))
    T, N_ROWS, H, _, W, C = videos.shape
    videos = videos.reshape((T, N_ROWS * H, num_per_row * W, C))
    return videos
