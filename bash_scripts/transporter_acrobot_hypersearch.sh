#!/usr/bin/bash


n_hidden=(128 256)
n_kpts=(3 4 5)
gmap_std=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
scheduler_epochs=(30 40 50 60)
scheduler_gamma=(0.5 0.75)
ct=0

for steps in ${scheduler_epochs[@]} ; do
  for gamma in ${scheduler_gamma[@]} ; do
    for hidden in ${n_hidden[@]} ; do
      for kpts in ${n_kpts[@]} ; do
        for g_std in ${gmap_std[@]} ; do
            yq e -i ".model.hidden_dim=${hidden}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
            yq e -i ".model.num_keypoints=${kpts}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
            yq e -i ".model.gaussian_map_std=${g_std}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
            yq e -i ".training.lr_scheduler_epoch_steps=${steps}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
            yq e -i ".training.lr_scheduler_gamma=${gamma}" /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml
            python /home/yannik/vssil/scripts/transporter_acrobot_training.py\
              -c /home/yannik/vssil/src/configs/transporter/transporter_acrobot_tmp.yml\
              -d /home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz\
              -i $ct
            ((ct++))
          done
        done
      done
    done
  done