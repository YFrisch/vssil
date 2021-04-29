# Visual Self-Supervised Imitation-Learning (VSSIL)

### SETUP:

* Install requirements from requirements.txt
* MIME-Dataset for "Stir" task: https://www.dropbox.com/sh/kfxdbxy1dsju79i/AAD_0YyCm17oradIgtoTEidja?dl=0%3E2.zip
    * Download and unpack to data/dataset/mime_stir
  
### RUN:
* Example for MIME Dataset loader: ```python tests/mime_dataset_test.py```
* Example for Deep Spatial Auto-Encoder: ```python tests/spatial_ae_test.py```
* View results and training process using ```tensorboard --logdir results/```

### TODOs: