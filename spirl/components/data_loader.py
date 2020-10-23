import glob
import imp
import os
import random
import h5py
import numpy as np
import torch.utils.data as data
import itertools

from spirl.utils.general_utils import AttrDict, map_dict, maybe_retrieve, shuffle_with_seed
from spirl.utils.pytorch_utils import RepeatedDataLoader
from spirl.utils.video_utils import resize_video


class Dataset(data.Dataset):

    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.dataset_size = dataset_size
        self.device = data_conf.device

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.samples_per_file = self._get_samples_per_file(self.filenames[0])

        self.shuffle = shuffle and phase == 'train'
        self.n_worker = 8 if shuffle else 1  # was 4 before

    def get_data_loader(self, batch_size, n_repeat):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
        return RepeatedDataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True, n_repeat=n_repeat, pin_memory=self.device == 'cuda',
                                  worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))

    def __getitem__(self, index):
        """Load a single sequence from disk according to index."""
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def _get_samples_per_file(self, path):
        """Returns number of data samples per data file."""
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def _get_filenames(self):
        """Loads filenames from self.data_dir, expects subfolders train/val/test, each with hdf5 files"""
        filenames = sorted(glob.glob(os.path.join(self.data_dir, self.phase + '/*.h5')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        return filenames

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return len(self.filenames) * self.samples_per_file


class GlobalSplitDataset(Dataset):
    """Splits in train/val/test using global percentages."""
    def _get_filenames(self):
        filenames = self._load_h5_files(self.data_dir)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        filenames = self._split_with_percentage(self.spec.split, filenames)
        return filenames

    def _load_h5_files(self, dir):
        filenames = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".h5"): filenames.append(os.path.join(root, file))
        return filenames

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['test']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]


class VideoDataset(Dataset):
    """Generic video dataset. Assumes that HDF5 file has images/states/actions/pad_mask."""
    def __init__(self, *args, resolution, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomize_length = self.spec.randomize_length if 'randomize_length' in self.spec else False
        self.crop_subseq = 'crop_rand_subseq' in self.spec and self.spec.crop_rand_subseq
        self.img_sz = resolution
        self.subsampler = self._get_subsampler()

    def __getitem__(self, index):
        data = self._get_raw_data(index)

        # maybe subsample seqs
        if self.subsampler is not None:
            data = self._subsample_data(data)

        # sample random subsequence of fixed length
        if self.crop_subseq:
            end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
            data = self._crop_rand_subseq(data, end_ind, length=self.spec.subseq_len)

        # Make length consistent
        start_ind = 0
        end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0) \
            if self.randomize_length or self.crop_subseq else self.spec.max_seq_len - 1
        end_ind, data = self._sample_max_len_video(data, end_ind, target_len=self.spec.subseq_len if self.crop_subseq
                                                                                  else self.spec.max_seq_len)

        if self.randomize_length:
            end_ind = self._randomize_length(start_ind, end_ind, data)
            data.start_ind, data.end_ind = start_ind, end_ind

        # perform final processing on data
        data.images = self._preprocess_images(data.images)

        return data

    def _get_raw_data(self, index):
        data = AttrDict()
        file_index = index // self.samples_per_file
        path = self.filenames[file_index]

        try:
            with h5py.File(path, 'r') as F:
                ex_index = index % self.samples_per_file  # get the index
                key = 'traj{}'.format(ex_index)

                # Fetch data into a dict
                data.images = F[key + '/images'][()]
                for name in F[key].keys():
                    if name in ['states', 'actions', 'pad_mask']:
                        data[name] = F[key + '/' + name][()].astype(np.float32)
        except:
            raise ValueError("Could not load from file {}".format(path))
        return data

    def _get_samples_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'].value

    def _get_subsampler(self):
        subsampler_class = maybe_retrieve(self.spec, 'subsampler')
        if subsampler_class is not None:
            subsample_args = maybe_retrieve(self.spec, 'subsample_args')
            assert subsample_args is not None  # need to specify subsampler args dict
            subsampler = subsampler_class(**subsample_args)
        else:
            subsampler = None
        return subsampler

    def _subsample_data(self, data_dict):
        idxs = None
        for key in data_dict:
            data_dict[key], idxs = self.subsampler(data_dict[key], idxs=idxs)
        return data_dict

    def _crop_rand_subseq(self, data, end_ind, length):
        """Crops a random subseq of specified length from the full sequence."""
        assert length <= end_ind + 1     # sequence needs to be longer than desired subsequence length
        start = np.random.randint(0, end_ind - length + 2)
        for key in data:
            data[key] = data[key][start : int(start+length)]
        return data

    def _sample_max_len_video(self, data_dict, end_ind, target_len):
        """ This function processes data tensors so as to have length equal to target_len
        by sampling / padding if necessary """
        extra_length = (end_ind + 1) - target_len
        if self.phase == 'train':
            offset = max(0, int(np.random.rand() * (extra_length + 1)))
        else:
            offset = 0

        data_dict = map_dict(lambda tensor: self._maybe_pad(tensor, offset, target_len), data_dict)
        if 'actions' in data_dict:
            data_dict.actions = data_dict.actions[:-1]
        end_ind = min(end_ind - offset, target_len - 1)

        return end_ind, data_dict

    @staticmethod
    def _maybe_pad(val, offset, target_length):
        """Pads / crops sequence to desired length."""
        val = val[offset:]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            return np.concatenate((val, np.zeros([int(target_length - len)] + list(val.shape[1:]), dtype=val.dtype)))
        else:
            return val

    def _randomize_length(self, start_ind, end_ind, data_dict):
        """ This function samples part of the input tensors so that the length of the result
        is uniform between 1 and max """

        length = 3 + int(np.random.rand() * (end_ind - 2))  # The length of the seq is from 2 to total length
        chop_length = int(np.random.rand() * (end_ind + 1 - length))  # from 0 to the reminder
        end_ind = length - 1
        pad_mask = np.logical_and((np.arange(self.spec['max_seq_len']) <= end_ind),
                                  (np.arange(self.spec['max_seq_len']) >= start_ind)).astype(np.float32)

        # Chop off the beginning of the arrays
        def pad(array):
            array = np.concatenate([array[chop_length:], np.repeat(array[-1:], chop_length, 0)], 0)
            array[end_ind + 1:] = 0
            return array

        for key in filter(lambda key: key != 'pad_mask', data_dict):
            data_dict[key] = pad(data_dict[key])
        data_dict.pad_mask = pad_mask

        return end_ind

    def _preprocess_images(self, images):
        assert images.dtype == np.uint8, 'image need to be uint8!'
        images = resize_video(images, (self.img_sz, self.img_sz))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        return images


class PreloadVideoDataset(VideoDataset):
    """Loads all sequences into memory for accelerated training (only possible for small datasets)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self._load_data()

    def _load_data(self):
        """Load all sequences into memory."""
        print("Preloading all sequences from {}".format(self.data_dir))
        return [super(PreloadVideoDataset, self)._get_raw_data(i) for i in range(len(self.filenames))]

    def _get_raw_data(self, index):
        return self._data[index]


class GlobalSplitVideoDataset(VideoDataset, GlobalSplitDataset):
    pass


class PreloadGlobalSplitVideoDataset(PreloadVideoDataset, GlobalSplitDataset):
    pass


class GlobalSplitStateSequenceDataset(GlobalSplitVideoDataset):
    """Outputs observation in data dict, not images."""
    def __getitem__(self, item):
        data = super().__getitem__(item)
        data.observations = data.pop('states')
        return data


class GlobalSplitActionSequenceDataset(GlobalSplitVideoDataset):
    """Outputs observation in data dict, not images."""
    def __getitem__(self, item):
        data = super().__getitem__(item)
        data.observations = data.pop('actions')
        return data


class MixedVideoDataset(GlobalSplitVideoDataset):
    """Loads filenames from multiple directories and merges them with percentage."""
    def _load_h5_files(self, unused_dir):
        assert 'data_dirs' in self.spec and 'percentages' in self.spec
        assert np.sum(self.spec.percentages) == 1
        files = [super(MixedVideoDataset, self)._load_h5_files(dir) for dir in self.spec.data_dirs]
        files = [shuffle_with_seed(f) for f in files]
        total_size = min([1 / p * len(f) for p, f in zip(self.spec.percentages, files)])
        filenames = list(itertools.chain.from_iterable(
            [f[:int(total_size*p)] for p, f in zip(self.spec.percentages, files)]))
        return filenames


class GeneratedVideoDataset(VideoDataset):
    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size

        if self.phase == 'train':
            return 10000
        else:
            return 200

    def _get_filenames(self):
        return [None]

    def _get_samples_per_file(self, path):
        pass

    def get_sample(self):
        raise NotImplementedError("Needs to be implemented by child class.")

    def __getitem__(self, index):
        if not self.shuffle:
            # Set seed such that validation is always deterministic
            np.random.seed(index)
        data = self.get_sample()
        return data

    @staticmethod
    def visualize(*args, **kwargs):
        """Enables dataset-specific visualization."""
        pass


class RandomVideoDataset(GeneratedVideoDataset):
    def get_sample(self):
        data_dict = AttrDict()
        data_dict.images = np.random.rand(self.spec['max_seq_len'], 3, self.img_sz, self.img_sz).astype(np.float32)
        data_dict.states = np.random.rand(self.spec['max_seq_len'], self.spec['state_dim']).astype(np.float32)
        data_dict.actions = np.random.rand(self.spec['max_seq_len'] - 1, self.spec['n_actions']).astype(np.float32)

        return data_dict
