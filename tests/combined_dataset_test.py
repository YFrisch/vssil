import os
from os.path import join

from src.data.utils import combine_mime_hd_kinect_tasks

cwd = os.getcwd()

data_set = combine_mime_hd_kinect_tasks(
    task_list=['stir', 'pour', 'stack', 'pass', 'place_in_box'],
    base_path=join(cwd, 'datasets/'),
    timesteps_per_sample=8,
    overlap=0,
    img_scale_factor=(160/240, 160/640)
)

print(len(data_set))
