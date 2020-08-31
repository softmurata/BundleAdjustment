# Globals
USE_SHAPENET=True # Set to True if training/testing on shapenet.

USE_AVERAGE=False # Set to True if training/testing the model that combines views with averaging.
ROTATE_BOX=False  # Set to True if want to train the latent 3D version.

SMALL_DECODER=False # Set to True if want to train/test the smaller decoder.
SMALL_DECODER_3D=False # Set to True if want to train/test the latent 3D version.

INPUT_ANGLES = True # Set to True if want to train/test with inputting input angles. This was found to be important for obtaining good results. 

# losses
MEAN_DEPTH=True # Set to True to subtract off the mean depth.
WEIGHT_MINMAX=True # Set to True to use a hinge loss around the object boundary.
