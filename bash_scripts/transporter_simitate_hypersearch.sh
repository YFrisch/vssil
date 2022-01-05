#!/usr/bin/bash

n_frames=(20)
n_kpts=(2 4 8)
gmap_std=(0.05 0.1 0.15 0.2 0.25)
ct=0

for frames in ${n_frames[@]} ; do
  for kpts in ${n_kpts[@]} ; do
    for g_std in ${gmap_std[@]} ; do
        yq e -i ".model.n_frames=${frames}" /home/yannik/vssil/src/configs/transporter/transporter_simitate_tmp.yml
        yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_simitate_tmp.yml
        yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_simitate_tmp.yml
        python /home/yannik/vssil/scripts/transporter_simitate_training.py\
          -c /home/yannik/vssil/src/configs/transporter/transporter_simitate_tmp.yml\
          -d /media/yannik/samsung_ssd/data/simitate_basic_motions_processed_64pix/\
          -i $ct
        ((ct++))
      done
    done
  done