#!/usr/bin/bash

n_frames=(2 3 4 5)
n_kpts=(4 8 16 32)
gmap_std=(0.001 0.050 0.075 0.100 0.125 0.150 0.175 0.5)
ct=0

for frames in ${n_frames[@]} ; do
  for kpts in ${n_kpts[@]} ; do
    for g_std in ${gmap_std[@]} ; do
        yq e -i ".model.n_frames=${frames}" /home/yannik/vssil/src/configs/transporter/transporter_human36m_tmp.yml
        yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_human36m_tmp.yml
        yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_human36m_tmp.yml
        python /home/yannik/vssil/scripts/transporter_human36m_training.py\
          -c /home/yannik/vssil/src/configs/transporter/transporter_human36m_tmp.yml\
          -d /media/yannik/samsung_ssd/data/human_36m_processed_128pix/\
          -i $ct
        ((ct++))
      done
    done
  done