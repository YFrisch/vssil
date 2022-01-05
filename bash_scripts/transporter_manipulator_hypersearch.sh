#!/usr/bin/bash

n_frames=(2 4 8)
n_kpts=(2 4 8)
gmap_std=(0.075 0.1 0.125 0.15 0.175)
ct=0

for frames in ${n_frames[@]} ; do
  for kpts in ${n_kpts[@]} ; do
    for g_std in ${gmap_std[@]} ; do
        yq e -i ".model.n_frames=${frames}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        python /home/yannik/vssil/scripts/transporter_manipulator_training.py\
          -c /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml\
          -d /media/yannik/samsung_ssd/data/dmcs_processed_64pix/\
          -i $ct
        ((ct++))
      done
    done
  done