import os
import numpy as np
from PIL import Image
from torchvision.transforms import Resize


def ch_first2last(video):
    return video.transpose((0,2,3,1))


def ch_last2first(video):
    return video.transpose((0,3,1,2))
    

def resize_video(video, size):
    if video.shape[1] == 3:
        video = np.transpose(video, (0,2,3,1))
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)
    return transformed_video


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


def create_video_grid(col_and_row_frames):
    video_grid_frames = np.concatenate([
        np.concatenate(row_frames, axis=-2)
        for row_frames in col_and_row_frames
    ], axis=-3)

    return video_grid_frames


