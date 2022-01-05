#!/usr/bin/bash

#n_frames=(2 3 4 5)
#n_kpts=(2 3 4 5)
n_frames=(2)
n_kpts=(4)
gmap_std=(0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15)
ct=0

for frames in ${n_frames[@]} ; do
  for kpts in ${n_kpts[@]} ; do
    for g_std in ${gmap_std[@]} ; do
        yq e -i ".model.n_frames=${frames}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
        yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
        yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
        python /home/yannik/vssil/scripts/transporter_acrobot_training.py\
          -c /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml\
          -d /home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz\
          -i $ct
        ((ct++))
      done
    done
  done