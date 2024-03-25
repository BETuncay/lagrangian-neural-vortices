import tensorflow as tf
import tensorflow.keras as keras
from lcs_unet import tversky_loss, focal_tversky_loss, dice_coef_loss, dice_coef
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from lcs_unet import create_unet2D, create_unet3D
from get_training_data import load, DataGenerator
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.python.framework.config import set_memory_growth


###
# Main Function used to train new networks
# Input Type, Hyperparameters, etc need to be set

## Hyperparameters
unet_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks'
custom_objects={'tversky_loss': tversky_loss, 'focal_tversky_loss': focal_tversky_loss,'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}
spatial_dimension = [512, 512]
filters = 16
n_channels = 12
batch_size = 4
dropout = 0
depth = 3
batch_norm=True
loss = 'Focal_Tversky_Loss_03_07_13'#'Focal_tversky_alpha_05_beta_05_gamma_13'
velocity_channels = 2
input_type = 'Velocity_Field_and_Flow_Map_2'

unet_name = f'3u_net2D_{input_type}_VC{velocity_channels}_C{n_channels}_F{filters}_D{depth}_L_{loss}_BN_DP_{dropout*100}_Batch_{batch_size}_SAMENORM' + '.h5'
unet_path = f'{unet_path}\{unet_name}'
logdir = "logs/" + unet_name

try:
    # load model if it exists 
    u_net = keras.models.load_model(unet_path, custom_objects=custom_objects)
except:
    # define network architecture

    print('Model not found! Creating new model!')

    # flow map input 
    u_net = create_unet2D(input_shape=(*spatial_dimension, n_channels), conv_shape=(3,3), filters=filters, pool_shape=(2,2), dropout=dropout, padding='valid', periodic=True, batch_norm=batch_norm, depth=depth, num_classes=1) 
    

# load training data
training_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set'
validation_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Validation_Data_Set'


# set generators
train_generator = DataGenerator(path=training_data_path, labels='training', batch_size=batch_size, dim=(512,512), n_channels=n_channels, input_type=input_type, batch_normalize= True, velocity_channels=velocity_channels)
validation_generator = DataGenerator(path=validation_data_path, labels='validation', batch_size=batch_size, dim=(512,512), n_channels=n_channels, input_type=input_type, batch_normalize= True, velocity_channels=velocity_channels)

# set callbacks
callbacks = [
    EarlyStopping(monitor='loss', verbose=1, patience=10, mode='min', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint(unet_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max'),
    TensorBoard(log_dir=logdir)
]

# start traing
u_net.fit(train_generator, epochs=100, validation_data=validation_generator, callbacks=callbacks, verbose=1)