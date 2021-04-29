# Visual Self-Supervised Imitation-Learning (VSSIL)

### SETUP:

* Make sure pip is installed
* Install external packages using ```pip install -r requirements.txt```  
* Install local packages with ```pip install -e .```
* MIME-Dataset for "Stir" task: https://www.dropbox.com/sh/kfxdbxy1dsju79i/AAD_0YyCm17oradIgtoTEidja?dl=0%3E2.zip
    * Download and unpack to data/dataset/mime_stir
  
### RUN:
* Example for MIME Dataset loader: ```python tests/mime_dataset_test.py```
* Example for the training of the Deep Spatial Auto-Encoder: ```python tests/spatial_ae_test.py```
* Example for inference of the DSAE: ```python tests/spatial_ae_test2.py```
* View results and training process using ```tensorboard --logdir results/```

### TODOs: