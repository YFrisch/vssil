from os import listdir
from os.path import isdir, join

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MimeBase(Dataset):

    """ Base wrapper for the MIME dataset."""

    def __init__(self,
                 sample_file_name: str = None,
                 base_path: str = '/home/yannik/vssil/data/datasets',
                 task: str = 'stir',
                 start_ind: int = 0,
                 stop_ind: int = -1,
                 timesteps_per_sample: int = -1,
                 overlap: int = 20
                 ):
        """ Creates class instance.

        :param base_path: Path to the super folder of all MIME tasks
        :param task: Specific MIME task. Please combine multiple tasks using the PyTorch api
        :param start_ind: Trajectory index to start sampling from (Between 0 and -2)
        :param stop_ind: Maximal trajectory index to sample from (Between 1 and -1)
        :param timesteps_per_sample: How many consec. frames to return per sample. Set -1 for whole trajectory.
        :param overlap: Overlap of the sample windows. Please see make_indices() for details.
        """
        super(MimeBase, self).__init__()

        assert (0 <= start_ind < stop_ind) or stop_ind == -1

        self.base_path = join(base_path, 'mime_' + task)
        self.base_paths = listdir(self.base_path)[start_ind:stop_ind]

        assert isdir(self.base_path), f"Can not read {self.base_path}"

        self.sample_paths, self.sample_lengths = self.gather_samples(sample_file_name)

        self.index_tuples = self.make_indices(lengths=self.sample_lengths,
                                              interval_size=timesteps_per_sample,
                                              overlap=overlap)

    def read_sample(self, path: str) -> (torch.Tensor, int):
        """ Function returning sample at given path, and length of that sample (# time-steps)."""
        raise NotImplementedError

    def gather_samples(self, sample_file_name: str):
        """ Reads all file paths and sample lengths for files at self.base_path with the given name.

        :param sample_file_name: File name for samples
        """

        print("##### Gathering sample paths and lengths.")

        sample_paths, sample_lengths = [], []
        for base_path in tqdm(self.base_paths):
            for file_name in listdir(join(self.base_path, base_path)):
                if file_name == sample_file_name:
                    sample_path = join(self.base_path, base_path, file_name)
                    sample_paths.append(sample_path)
                    sample_lengths.append(self.read_sample(sample_path)[1])  #
                else:
                    pass

        return sample_paths, sample_lengths

    def make_indices(self, lengths: list, interval_size: int, overlap: int):
        """ For a given list of (video) lengths, a given window size for samples,
                 and a given overlap for these samples, returns a list of index tuples to use during sampling.

                 E.g. if a video has 120 frames, a sample should contain 60 frames, and overlap is 30,
                  then we get 3 samples from that video: [0:60],[30:90],[60:120].

                :param lengths: List of (video) lengths
                :param interval_size: Size of the frame windows to sample
                :param overlap: Overlap of the frame windows (left and right)
                """

        print("##### Calculating index tuples.")

        assert interval_size > overlap, "Window size has to be greater than their overlap..."

        if interval_size == -1:  # Sample the whole video / trajectory
            return [(i, 0, -1) for i, l in enumerate(lengths)]

        indices = []
        for i, l in enumerate(lengths):
            if l < interval_size:
                raise NotImplementedError("The chosen sample window [i:i + timesteps_per_sample] exceeds the length of"
                                          "one of the samples."
                                          " Please choose a smaller window or -1 for whole trajectories.")
            start = 0
            stop = interval_size
            while stop <= l:
                indices.append((i, start, stop))
                start += interval_size - overlap
                stop += interval_size - overlap

        return indices

    def __len__(self):
        return len(self.index_tuples)

    def __getitem__(self, index):
        """ Returns sample of shape
            (T, C, H, W) of the given index.

        :param index: Sample index
        :return: Sample in (T, C, H, W) format
        """
        (sample_index, frame_start_index, frame_stop_index) = self.index_tuples[index]

        data, _ = self.read_sample(self.sample_paths[sample_index])
        data = data[frame_start_index:frame_stop_index]

        assert data.dim() == 4

        return data
