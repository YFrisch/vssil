# Visual Self-Supervised Imitation-Learning (VSSIL)

### SETUP:

* Make sure pip is installed
* Install external packages using ```pip install -r requirements.txt```  
* Install local packages with ```pip install -e .```
* Alternatively create conda env with ```environment.yml

  
### RUN:
* Example for loading the (processed) MIME data: ```python tests/video_dataset_test.py```
  * Also includes an example for data-augmentation
* Example for the training of the Deep Spatial Auto-Encoder: ```python scripts/spatial_ae_training.py```
* View results and training process using ```tensorboard --logdir results/```
* The baseline from "Unsupervised Learning of Object Structure and Dynamics" can be run with different
scripts, e.g. ```python scripts/ulosd_acrobot_training``` on the acrobot example data. This script takes two input arguents:
  * -c : The path to the configuration .yaml file
  * -d : The path to the video frame dataset, as specified in https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
