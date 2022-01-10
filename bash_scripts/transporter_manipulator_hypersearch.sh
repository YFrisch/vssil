#!/usr/bin/bash

n_hidden=(64 128 256)
n_kpts=(4 5 6)
gmap_std=(0.1 0.2 0.3 0.4 0.5)
ct=0

for hidden in ${n_hidden[@]} ; do
  for kpts in ${n_kpts[@]} ; do
    for g_std in ${gmap_std[@]} ; do
        yq e -i ".model.hidden_dim=${hidden}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml
        python /home/yannik/vssil/scripts/transporter_manipulator_training.py\
          -c /home/yannik/vssil/src/configs/transporter/transporter_manipulator_tmp.yml\
          -d /media/yannik/samsung_ssd/data/manipulator_processed_128pix/\
          -i $ct
        ((ct++))
      done
    done
  done