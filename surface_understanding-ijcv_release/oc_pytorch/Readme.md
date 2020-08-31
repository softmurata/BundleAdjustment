# Readme.md

This is the code for sidenet.

<h1>Run testing code</h1>
To run the testing code: download the models and data and put the models in folder ./models/ (or so it matches the path in ./launch_test.sh).
These models correspond to table 5 in the paper (e.g. trained with all forms of data augmentation + additional data).
The data should go in the path corresponding to that given in launch_test.sh

<h1>Train a model</h1>
The code is configured currently to train/test on the sculptures with the given sculptures (e.g. folder img_dataset_120c_fixedangle). However, by modifying in dataset.py, new folders with different renderings + objects can be added to train or test.
As is shown in launnch_test.sh, a given model can be tested with 1/2/3 views.

<h2>Edit Configurations</h2>
To change the model type and training configuration, edit utils/Globals.py:
- USE_SHAPENET: whether training/testing on shapenet
- USE_AVERAGE: whether to use max/average pooling
- ROTATE_BOX: whether to directly rotate the voxels or to project along projection rays (a la Yan et al, Nips 2017)

Decoders: if both are false, uses the standard one
- SMALL_DECODER: a smaller 57x57 decoder
- SMALL_DECODER_3D: whether to generate a latent 3D voxels

- INPUT_ANGLES: whether to make a viewpoint-dependent representation.


Loss functions:
- Whether to subtract the mean depth
- Whether to reweight


<h1>Requirements:</h1>
The models are evaluated with:
pytorch=0.4.0a0+2df578a
torchvision=0.2.0.

Pytorch was compiled from source on my machine, so it may have slight differences from the standard released versions.