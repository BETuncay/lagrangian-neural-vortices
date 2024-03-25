from itertools import count
from numpy.core.defchararray import translate
from tensorflow import keras
from tensorflow.keras.metrics import (BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall, AUC)
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.state_ops import batch_scatter_update
from get_training_data import load, load_all, load_specific, DataGenerator
from lcs_unet import tversky_loss, focal_tversky_loss, dice_coef_loss, dice_coef
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import colors
import time

## Plot the predictions of several networks
# compares networks contained in dictionaries (examples are below) 

def compare_masks(y_test, prediction):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(y_test[0,:,:], interpolation='bilinear', cmap=cm.Greys)
    ax[0].set_title('Ground Truth')

    ax[1].imshow(predictions[0,:,:], interpolation='bilinear', cmap=cm.Greys)
    ax[1].set_title('Prediction')

def compare_masks_indepth(x, y, vorticity, flow_field, lcs_mask, prediction):
    fig, ax = plt.subplots(2,2)
    fig.suptitle('hello :)', fontsize=16)

    ax[0,1].imshow(lcs_mask[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[0,1].set_title('Ground Truth')

    ax[0,0].streamplot(x, y, flow_field[0,:,:,0], flow_field[0,:,:,1], density = 0.5)
    ax[0,0].set_title('Flow Field')

    ax[1,0].imshow(vorticity, interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[1,0].set_title('Vorticity')

    ax[1,1].imshow(prediction[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[1,1].set_title('Prediction')

def compare_results(vorticity, lcs_mask, prediction, network_count, flow_count, network_names, color_error=True, metrics=None):
    
    cmap = cm.Greys
    norm = None
    if color_error:
        cmap = colors.ListedColormap(['white', 'red', 'blue', 'black'])
        bounds=[0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
    fig = plt.figure()
    gs = gridspec.GridSpec(flow_count, network_count + 2)
    gs.update(wspace=0, hspace=0.1)

    for flow in range(flow_count):
        for net in range(network_count):

            if flow == 0:
                ax = plt.subplot(gs[flow, net], title=network_names[net])
            else:
                ax = plt.subplot(gs[flow, net])

            if color_error:
                prediction[net,flow,:,:] = np.rint(prediction[net,flow,:,:]) + lcs_mask[flow,:,:] * 2
 
            ax.imshow(prediction[net,flow,:,:], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

            if flow == flow_count - 1 and  np.any(metrics):
                network_summary = f'''Loss ='{metrics[net,0,0]}\u00B1{metrics[net,0,1]}\nDSC ='{metrics[net,1,0]}\u00B1{metrics[net,1,1]}\nPrecision ='{metrics[net,2,0]}\u00B1{metrics[net,2,1]}\nRecall ='{metrics[net,3,0]}\u00B1{metrics[net,3,1]}'''
                ax.set_xlabel(network_summary)
                print(network_names[net])
                print(network_summary)


        ax1 = plt.subplot(gs[flow, network_count])
        ax2 = plt.subplot(gs[flow, network_count + 1])
        ax1.imshow(lcs_mask[flow,:,:], interpolation='nearest', origin='lower', cmap=cm.Greys)
        ax2.imshow(vorticity[flow,:,:], interpolation='nearest', origin='lower', cmap=cm.Greys)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        if flow == 0:
            ax1.set_title('Ground Truth')
            ax2.set_title('Absolute Vorticity')

    plt.show()
    print('end')

def evaluate_network(u_net, data_loader):
    
    data_set = glob.glob(test_data_path + "/*.mat")
    amount = len(data_set)
    metrics = np.empty([amount, 8])

    batch_size = data_loader.batch_size
    data_loader.batch_size = 1
    for i in range(amount):
        X, Y = data_loader.get_item([data_set[i]])
        print(data_set[i])
        results = u_net.evaluate(X, Y)
        metrics[i] = results

    mean_loss = np.sum(metrics[:,0]) / amount
    mean_dice = np.sum(metrics[:,1]) / amount
    mean_precision = np.sum(metrics[:,2]) / amount
    mean_recall = np.sum(metrics[:,3]) / amount

    std_loss = np.sqrt((1/(amount)) * np.sum(np.power(metrics[:,0] - mean_loss,2)))
    std_dice = np.sqrt((1/(amount)) * np.sum(np.power(metrics[:,1] - mean_dice,2)))
    std_precision = np.sqrt((1/(amount)) * np.sum(np.power(metrics[:,2] - mean_precision,2)))
    std_recall = np.sqrt((1/(amount)) * np.sum(np.power(metrics[:,3] - mean_recall,2)))

    results = np.array([[mean_loss, std_loss],[mean_dice, std_dice],[mean_precision, std_precision],[mean_recall, std_recall]])
    results = np.round(results,3)
    data_loader.batch_size = batch_size
    #precision = np.sum(metrics_dict['tp']) / (np.sum(metrics_dict['tp']) + np.sum(metrics_dict['fp']))
    #recall = np.sum(metrics_dict['tp']) / (np.sum(metrics_dict['tp']) + np.sum(metrics_dict['fn']))
    #dsc = 2*np.sum(metrics_dict['tp']) / (2*np.sum(metrics_dict['tp']) + np.sum(metrics_dict['fp']) + np.sum(metrics_dict['fn']))
    #tversky = tp / (2*tp + 0.3*fp + 0.7*fn)
    
    return results

def compare_results_color_test():
    
    cmap = colors.ListedColormap(['white', 'red', 'blue', 'black'])
    bounds=[0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N) 
        

    prediciton = np.array([[1,1],[0,0]])
    ground_truth = np.array([[1,0],[1,0]])
    fig = plt.figure()

    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=0.2, hspace=0.2)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])

    ax1.imshow(prediciton, interpolation='nearest', cmap=cm.Greys)
    ax2.imshow(ground_truth, interpolation='nearest', cmap=cm.Greys)

    diff = prediciton + ground_truth * 2
    ax3.imshow(diff, interpolation='nearest', cmap=cm.Greys)
    ax4.imshow(diff, interpolation='nearest', cmap=cmap, norm=norm)

    plt.show()
    print('end')

# plots network predictions along with the ground truth and abs vorticity
# inputs:   data_dict: information about network (see examples),
#           flow_count: amount of flows to compare, randomly chosen from test_data_path
#           calculate_metrics: boolean if metrics should be calculated or not
def compare_networks(data_dict, flows_path, flow_count, calculate_metrics, random_flows=True):

    network_count = len(data_dict['network_names'])

    # load models
    network_list = []
    for i in range(network_count):
        path = f"{data_dict['network_path']}\\{data_dict['network_names'][i]}"
        u_net = keras.models.load_model(path, custom_objects=custom_objects)
        u_net.compile(optimizer=Adam(), loss=tversky_loss, metrics=METRICS)
        network_list.append(u_net)

    predictions = np.empty([network_count, flow_count, shape[0], shape[1]], dtype='float32')
    results = np.empty([network_count, 4, 2])

    if not random_flows: 
        flows_path_1 = ['D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0105_7_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0635_7_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_1085_3_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0530_3_3.mat']
        
        flows_path_2 = ['D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0465_7_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_1080_7_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0495_7_3.mat',
                    'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set\\lcs_0105_7_3.mat']
                    #
    # load test data
    velocity_field, _, _, _, full_dataset = load_all(flows_path, count=flow_count, shape=shape, channels=31, translate=False, shuffle=True)

   
    # 'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set_new\\lcs_0310_7_3.mat'
    train_generator = DataGenerator(path=flows_path, labels='training', batch_size=flow_count, dim=(512,512), translate=False, batch_normalize=True,
                                n_channels=4, velocity_channels=0, input_type='Flow_Map')


    # calculate vorticity
    vorticity = np.empty([flow_count, shape[0], shape[1]], dtype='float32')
    for c in range(flow_count):
        v_x = np.gradient(velocity_field[c,:,:,0])[0]
        u_y = np.gradient(velocity_field[c,:,:,1])[1]
        vorticity[c] = np.abs(v_x - u_y)

    # Generate predictions
    for net in range(network_count):
        train_generator.n_channels = data_dict['channels_list'][net]
        train_generator.input_type = data_dict['input_type_list'][net]
        train_generator.velocity_channels = data_dict['velocity_channels'][net]    
        X, Y = train_generator.get_item(full_dataset)

        if calculate_metrics:
            results[net] = evaluate_network(network_list[net], train_generator)
        predictions[net] = np.squeeze(network_list[net].predict(X))

    #if not calculate_metrics:
    #    results = None
    
    compare_results(vorticity, Y, predictions, network_count, flow_count, data_dict['network_titles'], metrics=results)
    
    plt.show()
    print('end')

def calc_execution_time(data_dict):
    path = f"{data_dict['network_path']}\\{data_dict['network_names'][0]}"
    u_net = keras.models.load_model(path, custom_objects=custom_objects)

    train_generator = DataGenerator(path=test_data_path, labels='training', batch_size=1, dim=(512,512), translate=False, batch_normalize=True,
                                n_channels=4, velocity_channels=0, input_type='Velocity_Field')

    train_generator.n_channels = data_dict['channels_list'][0]
    train_generator.input_type = data_dict['input_type_list'][0]
    train_generator.velocity_channels = data_dict['velocity_channels'][0]

    data_set = glob.glob(test_data_path + "/*.mat")
    amount = len(data_set)
    exec_time = np.empty([amount])

    
    # load keras first
    X, _ = train_generator.get_item([data_set[0]])
    prediction = np.squeeze(u_net.predict(X))

    for i in range(amount):
        start_time = time.time()
        X, _ = train_generator.get_item([data_set[i]])   
        prediction = np.squeeze(u_net.predict(X))
        elapsed = time.time() - start_time
        exec_time[i] = elapsed

    mean_time = np.sum(exec_time / amount)
    std_time = np.sqrt((1/amount) * np.sum(np.power(exec_time - mean_time, 2)))

    results = np.array([mean_time, std_time])
    results = np.round(results,3)

## global variables
test_data_path = 'D:\\_Tools_Data\\Matlab_Data\\lagrangian-neural-vortices\\Vortex_Extraction\\Results\\Validation_Data_Set'
custom_objects={'tversky_loss': tversky_loss, 'focal_tversky_loss': focal_tversky_loss,'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}
METRICS = [
    dice_coef,
    Precision(name='precision'),
    Recall(name='recall'),
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn')
]

shape = [512, 512]
channels = 10

# example data_dicts
compare_velocity_data = {

    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',

    'network_names':[r'u_net2D_Velocity_Field_C2_F16_D3_L_focal_taversky_BN.h5',
                     r'u_net2D_Velocity_Field_C6_F16_D3_L_focal_taversky_BN.h5', 
                     r'u_net2D_Velocity_Field_C10_F16_D3_L_dice_coef_BN_DP_0_Batch_4.h5', 
                     r'u_net2D_Velocity_Field_C50_F16_D3_L_focal_taversky_BN_DELETE.h5'
                     ],
    'network_titles':['1 timestep', '3 timestep', '5 timestep', '25 timestep'],
    'channels_list':[2, 6, 10, 50],
    'input_type_list':['Velocity_Field', 'Velocity_Field', 'Velocity_Field', 'Velocity_Field'],
    'velocity_channels':[0, 0, 0, 0]
}

compare_flow_map_variants = {

    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',

    'network_names':[r'u_net2D_Flow_Map_VC0_C4_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5',
                    r'u_net2D_Flow_Map_2_C2_F16_D3_L_focal_tverskey_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5',
                    r'DELETE_u_net2D_Flow_Map_VC0_C20_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5',
                    r'u_net2D_Flow_Map_2_step_VC0_C10_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5'
                    ],
    'network_titles':['Flow map', 'Displacement map', 'Trajectory', 'Trajectory displacement'],
    'channels_list':[4, 2, 20, 10],
    'input_type_list':['Flow_Map', 'Flow_Map_2', 'Flow_Map', 'Flow_Map_2_step'],
    'velocity_channels':[0, 0, 0, 0]
}

compare_velocity_sampled = {

    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',
    'network_names':[r'u_net2D_Velocity_Field_VC0_C10_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5',
                    r'u_net2D_Velocity_Field_TimeSampled_VC0_C10_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5'
                    ],
    'network_titles':['time_consecutive10', 'time sample10'],
    'channels_list':[10, 10],
    'input_type_list':['Velocity_Field', 'Velocity_Field_TimeSampled'],
    'velocity_channels':[0, 0]
}

compare_losses = {

    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',

    'network_names':[r'u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Cross_entrophy_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Dice_Coefficient_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Tversky_Loss_03_07_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Tversky_Loss_07_03_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_05_05_13_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'3u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_03_07_13_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_03_07_13_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_07_03_13_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_05_05_075_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_03_07_075_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_07_03_075_BN_DP_0_Batch_4_SAMENORM.h5',
                    ],
    'network_titles':['Cross Entrophy', 'Dice Loss', 'Tversky 0.3, 0.7', 'Tversky 0.7, 0.3',
                    'Focal Tversky 0.5, 0.5, 1.34', 'new Focal Tversky 0.3, 0.7, 1.34', '2.Focal Tversky 0.3, 0.7, 1.34',
                    'Focal Tversky 0.7, 0.3, 1.34', 'Focal Tversky 0.5, 0.5, 0.75', 'Focal Tversky 0.3, 0.7, 0.75',
                    'Focal Tversky 0.7, 0.3, 0.75'],
    'channels_list':[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    'input_type_list':['Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2',
                      'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2',
                      'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2',
                      'Velocity_Field_and_Flow_Map_2', 'Velocity_Field_and_Flow_Map_2'
                      ],
    'velocity_channels':[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}

best_net = {

    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',

    'network_names':[r'u_net2D_Flow_Map_2_step_VC0_C10_F16_D3_L_Focal_tversky_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5',
                    r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_05_05_075_BN_DP_0_Batch_4_SAMENORM.h5',
                    #r'2u_net2D_Velocity_Field_and_Flow_Map_2_VC2_C12_F16_D3_L_Focal_Tversky_Loss_03_07_13_BN_DP_0_Batch_4_SAMENORM.h5',
                    r'u_net2D_Velocity_Field_C6_F16_D3_L_focal_taversky_BN.h5'
                    ],
    'network_titles':['Trajectory Displacement','Velocity Field and Trajectory Displacement', 'Velocity Field'],
    'channels_list':[10, 12, 6],
    'input_type_list':['Flow_Map_2_step', 'Velocity_Field_and_Flow_Map_2', 'Velocity_Field'],
    'velocity_channels':[0, 2, 0]
}

extraction_time_test = {
    'network_path':r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\U-Net\trained_networks',
    'network_names':[r'u_net2D_Velocity_Field_C10_F16_D3_L_dice_coef_BN_DP_0_Batch_4.h5'],
    'network_titles':['extraction_time_test'],
    'channels_list':[10],
    'input_type_list':['Velocity_Field'],
    'velocity_channels':[0]
}

def main():

    flows_path = test_data_path
    flow_count = 4
    calculate_metrics = False
    random_flows = True
    compare_networks(compare_flow_map_variants, flows_path, flow_count, calculate_metrics, random_flows)
    #calc_execution_time(extraction_time_test)

main()

# u_net2D_Velocity_Field_and_Flow_Map_2_VC4_C8_F16_D3_L_focal_tverskey_alpha_05_beta_05_gamma_13_BN_DP_0_Batch_4.h5

