# Visual Self-Supervised Imitation-Learning (VSSIL)

### SETUP:

* Make sure pip is installed
* Install external packages using ```pip install -r requirements.txt```  
* Install local packages with ```pip install -e .```

  
### RUN:
* Example for loading the (processed) MIME data: ```python tests/video_dataset_test.py```
* Example for the training of the Deep Spatial Auto-Encoder: ```python tests/spatial_ae_test.py```
* Example for inference of the DSAE: ```python tests/spatial_ae_test2.py```
* View results and training process using ```tensorboard --logdir results/```

### TODOs: