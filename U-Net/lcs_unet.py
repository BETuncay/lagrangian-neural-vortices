import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPool2D, Conv3D, Conv3DTranspose, MaxPool3D, Dropout, Input, concatenate, Cropping2D, BatchNormalization, LeakyReLU, LayerNormalization)
from tensorflow.keras.metrics import (BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall, AUC)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


###
# Keras 2D U-Net model is defined in this file along with the corresponding operations
# (3D U-Net is deprecated)

# https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def conv2d_block(x, conv_shape = (3,3), filters=64, dropout=0, padding='valid', activation='relu', periodic=True, stride = (1,1), batch_norm=True):

    # first convolution 
    if periodic:
        x = periodic_padding_flexible(x, axis=(1,2),padding=(1,1))
    x = Conv2D(filters=filters, kernel_size=conv_shape, activation=activation, kernel_initializer='he_normal', padding=padding, strides=stride, data_format='channels_last')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)

    if dropout:
        x = Dropout(dropout)(x)

    # second convolution 
    if periodic:
        x = periodic_padding_flexible(x, axis=(1,2),padding=(1,1))
    x = Conv2D(filters=filters, kernel_size=conv_shape, activation=activation, kernel_initializer='he_normal', padding=padding, strides=stride, data_format='channels_last')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)


    if dropout:
        x = Dropout(dropout)(x)
    
    return x

def conv3d_block(x, conv_shape = (3,3,2), filters=64, dropout=0, padding='valid', activation='relu', periodic=True, stride=(1,1,1), batch_norm=True):

    # first convolution 
    if periodic:
        x = periodic_padding_flexible(x, axis=(1,2),padding=(1,1))
    x = Conv3D(filters=filters, kernel_size=conv_shape, activation=activation, kernel_initializer='he_normal', padding=padding, strides=stride, data_format='channels_last')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)

    if dropout:
        x = Dropout(dropout)(x)

    # second convolution 
    if periodic:
        x = periodic_padding_flexible(x, axis=(1,2),padding=(1,1))
    x = Conv3D(filters=filters, kernel_size=conv_shape, activation=activation, kernel_initializer='he_normal', padding=padding, strides=stride, data_format='channels_last')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    return x

def create_unet2D(input_shape, conv_shape = (3,3), filters = 16, pool_shape = (2,2), dropout = 0.0, padding='valid', depth = 4, num_classes=1, periodic=True, batch_norm=True):
    inputs = Input(shape=input_shape, dtype='float32')
    
    x = inputs

    #if batch_norm:
    #    x = BatchNormalization()(x)

    down_layers = []
    for i in range(depth):
        x = conv2d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)
        down_layers.append(x)
        x = MaxPool2D(pool_shape)(x)
        filters *= 2

    x = conv2d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)

    # expansive path
    for layer in reversed(down_layers):
        filters /= 2
        x = Conv2DTranspose(filters, pool_shape, strides=pool_shape, padding='same')(x)
        x = concatenate([x, layer])
        x = conv2d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)


    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    METRICS = [
      dice_coef,
      TruePositives(name='tp'),
      FalsePositives(name='fp'),
      TrueNegatives(name='tn'),
      FalseNegatives(name='fn'), 
      BinaryAccuracy(name='binary_accuracy'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc'),
      AUC(name='prc', curve='PR'),
    ]
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


    model.compile(optimizer=Adam(learning_rate=0.001), loss=focal_tversky_loss, metrics=METRICS)
    return model

def create_unet3D(input_shape, conv_shape = (3,3,2), filters = 16, pool_shape = (2,2,1), dropout = 0, padding='valid', depth = 4, num_classes=1, periodic=True, batch_norm=True):
    inputs = Input(shape=input_shape, dtype='float32')
    
    x = inputs
    down_layers = []
    for i in range(depth):
        x = conv3d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)
        down_layers.append(x)
        x = MaxPool3D(pool_shape)(x)
        filters *= 2

    x = conv3d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)

    # expansive path
    for layer in reversed(down_layers):
        filters /= 2
        x = Conv3DTranspose(filters, pool_shape, strides=pool_shape, padding='same')(x)
        x = concatenate([x, layer])
        x = conv3d_block(x=x, conv_shape=conv_shape, filters=filters, dropout=dropout, padding=padding, periodic=periodic, batch_norm=batch_norm)


    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(),  loss=tversky_loss, metrics=[dice_coef])
    return model

def test_net():

    u_net = create_unet3D(input_shape=(512, 512, 6, 2), conv_shape=(3,3,2), filters=16, pool_shape=(2,2,1), dropout=0, padding='valid', depth=1, num_classes=1, periodic=True, batch_norm=True) 
    u_net.summary()
    plot_model(u_net, to_file='3D_model.png', show_shapes=True, show_layer_names=True)

# Metrics / Loss functions
# inspired from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Tversky-Loss
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.dot(y_true_f, y_pred_f))
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1e-6, alpha=0.3, beta=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_pos = K.sum(y_pred_pos * (1 - y_true_pos))
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    return (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=1.333334):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)