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
from tqdm import tqdm
import pickle
import re, collections

class Dataset(data.Dataset):

    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.dataset_size = dataset_size
        self.device = data_conf.device

        print('loading files from', self.data_dir)
        
        # get both filenames, and quantized_list
        self.filenames, self.quantized_list = self._get_filenames()
        
        
        self.filenames = self._filter_filenames(self.filenames)
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
        # does not operate, since will be overwritted
        """Loads filenames from self.data_dir, expects subfolders train/val/test, each with hdf5 files"""
        filenames = sorted(glob.glob(os.path.join(self.data_dir, self.phase + '/*.h5')))

        
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        
        
        return filenames

    def _filter_filenames(self, filenames):
        """Optionally filters filenames / limits to max number of filenames etc."""
        
        if "n_seqs" in self.spec:
            # limit the max number of sequences in dataset
            if self.phase == "train" and len(filenames) < self.spec.n_seqs:
                raise ValueError("Not enough seqs in dataset!")
            filenames = filenames[:self.spec.n_seqs]
            

        if "seq_repeat" in self.spec:
            # repeat sequences in dataset
            repeat = max(self.spec.seq_repeat, self.dataset_size / len(filenames))
            filenames *= int(repeat)
            filenames = shuffle_with_seed(filenames)

        return filenames

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return len(self.filenames) * self.samples_per_file


class GlobalSplitDataset(Dataset):
    """Splits in train/val/test using global percentages."""
    def _get_filenames(self):
        
        # 87925
        filenames = self._load_h5_files(self.data_dir)
        
        with open("quantized_list_real.pickle","rb") as fr:
            quantized_list_real = pickle.load(fr)
        


        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        
        
        filenames = shuffle_with_seed(filenames)
        quantized_list_real = shuffle_with_seed(quantized_list_real)
        

        # give identical split to quantized data as well
        filenames = self._split_with_percentage(self.spec.split, filenames)
        quantized_list_real = self._split_with_percentage(self.spec.split, quantized_list_real)
        
        return filenames, quantized_list_real

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
        
        ## Load pickle
        with open("sorted_tokens_BPE_1000.pickle","rb") as fr:
            self.sorted_tokens_BPE_1000 = pickle.load(fr)

    def __getitem__(self, index):
        
        
        data = self._get_raw_data(index)
        
        
        quantized_data = self.quantized_list[index]
        
        #print(data.actions.shape)
        #print(len(quantized_data))
  
        
        # sample random subsequence of fixed length
        assert self.randomize_length is True
        

        # sample random subsequence of fixed length
        if self.randomize_length :
            end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
            
            
            # np.random.randint does not include last index
            start_idx = int(np.random.randint(0, data.actions.shape[0]-1, size=1))
   

            BPE_skill = self.BPE_function(quantized_data[start_idx:], self.sorted_tokens_BPE_1000)
            


            #print(data.actions.shape)
            #print(len(quantized_data))
            #print(quantized_data[start_idx:])
            #print(BPE_skill)
            #print(len(BPE_skill))

            
            if len(BPE_skill)==0 :
                BPE_skill_len = 1
                
            elif len(BPE_skill)>=self.spec.max_seq_len :
                BPE_skill_len = self.spec.max_seq_len
                
            else :
                BPE_skill_len = len(BPE_skill)
            
            data = self._crop_rand_subseq(data, end_ind, length=BPE_skill_len)
            
            
        # Make length consistent
        end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
        end_ind, data = self._sample_max_len_video(data, end_ind, target_len=self.spec.max_seq_len)



        # perform final processing on data
        data.images = self._preprocess_images(data.images)



        # index of padded actions
        # should be one smaller
        mask_seq = np.zeros((self.spec.max_seq_len, 2))
        
        # where values present
        mask_seq[:BPE_skill_len] = [1,1]
        
        data.action_mask = mask_seq

 

        return data



    def BPE_function(self, quantized_data, BPE_dict) :

        BPE_result = self.tokenize_word(string=quantized_data, sorted_tokens=BPE_dict, unknown_token='</u>')
        
        return BPE_result


    # tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>')
    def tokenize_word(self, string, sorted_tokens, unknown_token='</u>'):
        
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        
        # sorted_tokens : [..., '63', '36', '84', '4', '5', '8', '3', '2', '6', '7', '1']
        # sorted_tokens : ['newest</w>', 'widest</w>', 'lower</w>', 'low</w>']
        # sorted_tokens[1500:] = length13
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]

            # token_reg = re.escape(token.replace('.', '[.]'))
            

            # [(1, 2), (4, 5)]
            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token, string)]
            
            
            # 새로 입력받은 string과 sorted_tokens에 저장되어 있는 패턴들과 하나씩 대조한다
            if len(matched_positions) == 0:
                continue
            
            # [4]
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]
            
            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                
                # string의 시작 index와 찾은 pattern의 시작지점 사이의 substring에 대해서 재귀함수
                # pattern의 시작지점과, 뒤에 pattern의 시작지점 사이의 substring에 대해서 재귀함수
                # 이 때 각 재쉬함수들은 sorted_tokens[i+1:]을 사용한다. 즉 아직 대조하지 않은 pattern에 대해서만 실행
                substring = string[substring_start_position:substring_end_position]
            
                break
            break
   
        return substring




    def _get_raw_data(self, index):
        data = AttrDict()
        file_index = index // self.samples_per_file
        path = self.filenames[file_index]

        try:
            with h5py.File(path, 'r') as F:
                ex_index = index % self.samples_per_file  # get the index
                key = 'traj{}'.format(ex_index)

                # Fetch data into a dict
                for name in F[key].keys():
                    if name in ['states', 'actions', 'pad_mask']:
                        data[name] = F[key + '/' + name][()].astype(np.float32)

                if key + '/images' in F:
                    data.images = F[key + '/images'][()]
                else:
                    data.images = np.zeros((data.states.shape[0], 2, 2, 3), dtype=np.uint8)
        except:
            raise ValueError("Could not load from file {}".format(path))
        return data

    def _get_samples_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'][()]

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
