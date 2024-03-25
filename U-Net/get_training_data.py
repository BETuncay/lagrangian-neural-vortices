import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import glob
from tensorflow.keras.utils import Sequence
#import keras
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
from numpy import linalg as LA

## Dataloader Class
# The training data does not fully fit into memory 
# --> Use Keras DataGenerator to load different input types 



class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, path, labels, batch_size=32, dim=(512,512), n_channels=12,
                 n_classes=1, shuffle=True, translate=True, batch_normalize= True, input_type='Velocity_Field', dtype='float32', velocity_channels=0):
        'Initialization'
        self.file_list = []
        if type(path) == list:
            for p in path:
                self.file_list += (glob.glob(p + "/*.mat"))
        else:
            self.file_list = glob.glob(path + "/*.mat")

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels    
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.translate = translate
        self.batch_normalize = batch_normalize
        self.dtype = dtype
        self.input_type = input_type
        self.velocity_channels = velocity_channels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        file_list_temp = [self.file_list[k] for k in indexes]

        # Generate data
        if self.input_type == 'Velocity_Field':
            return self.__data_generation_velocity_field(file_list_temp)
        elif self.input_type == 'Flow_Map':
            return self.__data_generation_flow_map(file_list_temp)
        elif self.input_type == 'Flow_Map_Gradient':
            return self.__data_generation_flow_map_gradient(file_list_temp)
        elif self.input_type == 'Flow_Map_and_Gradient':
            return self.__data_generation_flow_map_and_gradient(file_list_temp)
        elif self.input_type == 'Velocity_Field_and_Gradient':
            return self.__data_generation_velocity_field_and_gradient(file_list_temp)
        elif self.input_type == 'Flow_Map_2':
            return self.__data_generation_flow_map_2(file_list_temp)
        elif self.input_type == 'Flow_Map_and_Gradient_2':
            return self.__data_generation_flow_map_and_gradient_2(file_list_temp)
        elif self.input_type == 'Velocity_Field_and_Flow_Map_2':
            return self.__data_generation_velocity_grid_and_flow_map(file_list_temp)
        elif self.input_type == 'CG_Tensor':
            return self.__data_generation_cg_tensor(file_list_temp)
        elif self.input_type == 'Reverse_Flow_Map_2':
            return self.__data_generation_reverse_flow_map_2(file_list_temp)
        elif self.input_type == 'Velocity_Field_TimeSampled':
            return self.__data_generation_velocity_field_time_sampled(file_list_temp)
        elif self.input_type == 'Flow_Map_2_step':
            return self.__data_generation_flow_map_2_step(file_list_temp)
        elif self.input_type == 'Velocity_Field_TimeSampled_and_Flow_Map_2':
            return self.__data_generation_velocity_field_time_sampled_and_flow_map(file_list_temp)
            
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# fur jeden input typen einzeln eine function schreiben mit schnittstelle mat_datei
# dann sind alle __data_generation_functionen viel besser geschrieben

    def __data_generation_velocity_field(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)
        #X = np.empty([self.batch_size, 512, 512, 12], dtype='float32')
        #Y = np.empty([self.batch_size, 512, 512], dtype='float32')

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)

            flow_field_u = mat_contents['flow_field_u']
            flow_field_v = mat_contents['flow_field_v']
            shape = np.shape(flow_field_u)
            
            flow_field = np.empty([self.n_channels, shape[1], shape[2]], dtype='float32')
            for j in range(self.n_channels // 2):
                flow_field[j * 2] = flow_field_u[j]
                flow_field[2 * j + 1] = flow_field_v[j]
            
            flow_field = np.transpose(flow_field, (1, 2, 0))
            lcs_mask = mat_contents['lcs_mask']

            # shape flow_field = (512, 512, 12) --> [0, 1] = [x, y]
            # shape lcs_mask = (512, 512) --> [0, 1] = [x, y]
            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                flow_field_shape = np.shape(flow_field)
                flow_field = flow_field.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_field = flow_field.reshape(flow_field_shape)
                flow_field = flow_field / max_norm
            X[i] = flow_field
            Y[i] = lcs_mask


        return X, Y

    def __data_generation_flow_map(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            trajectory_mat = mat_contents['Trajectory']
            shape = np.shape(trajectory_mat)
            flow_map = np.empty([shape[0], shape[1], self.n_channels], dtype='float32')

            take = np.round(np.linspace(0, shape[2] // 2 - 1, self.n_channels // 2)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                flow_map = translate_data(flow_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)

            if self.batch_normalize:

                flow_map_shape = np.shape(flow_map)
                flow_map = flow_map.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_map, flow_map)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_map = flow_map.reshape(flow_map_shape)
                flow_map = flow_map / max_norm

            X[i] = flow_map
            Y[i] = lcs_mask


        return X, Y

    def __data_generation_flow_map_2(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            
            trajectory_mat = mat_contents['Trajectory']
            shape = np.shape(trajectory_mat)
            flow_map = np.empty([shape[0], shape[1], self.n_channels + 2], dtype='float32')

            take = np.round(np.linspace(0, shape[2] // 2 - 1, self.n_channels // 2 + 1)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            diff_map = np.empty([shape[0], shape[1], self.n_channels], dtype='float32')
            for idx in range(self.n_channels // 2):
                diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,0:2]

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                diff_map_shape = np.shape(diff_map)
                diff_map = diff_map.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                max_norm = np.sqrt(np.amax(max_norm))

                diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm
            
            X[i] = diff_map
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_reverse_flow_map_2(self, file_list_temp):

        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            
            trajectory_mat = mat_contents['Trajectory']
            shape = np.shape(trajectory_mat)
            flow_map = np.empty([shape[0], shape[1], self.n_channels + 2], dtype='float32')

            take = np.round(np.linspace(0, shape[2] // 2 - 1, self.n_channels // 2 + 1)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            diff_map = np.empty([shape[0], shape[1], self.n_channels], dtype='float32')
            for idx in range(self.n_channels // 2):
                #diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,0:2]
                diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,-2:] - flow_map[:,:,2*idx:2*idx+2]
            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                diff_map_shape = np.shape(diff_map)
                diff_map = diff_map.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                max_norm = np.sqrt(np.amax(max_norm))

                diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm
            
            X[i] = diff_map
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_flow_map_and_gradient(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            

            flow_map_mat = mat_contents['Flow_Map']
            lcs_mask = mat_contents['lcs_mask']
            grad_f11 = mat_contents['grad_Flow_Map11']
            grad_f12 = mat_contents['grad_Flow_Map12']
            grad_f21 = mat_contents['grad_Flow_Map21']
            grad_f22 = mat_contents['grad_Flow_Map22']

            shape = [512, 512, 8]
            flow_map_and_gradient = np.empty([shape[0], shape[1], shape[2]], dtype='float32')
            flow_map_and_gradient[:,:,0:2] = flow_map_mat[0,:,:,:]
            flow_map_and_gradient[:,:,2:4] = flow_map_mat[1,:,:,:]
            
            flow_map_and_gradient[:,:,4] = grad_f11
            flow_map_and_gradient[:,:,5] = grad_f12
            flow_map_and_gradient[:,:,6] = grad_f21
            flow_map_and_gradient[:,:,7] = grad_f22

            lcs_mask = mat_contents['lcs_mask']

            # shape flow_field = (512, 512, 12) --> [0, 1] = [x, y]
            # shape lcs_mask = (512, 512) --> [0, 1] = [x, y]
            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                flow_map_and_gradient = translate_data(flow_map_and_gradient, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:
                scaler = StandardScaler()
                flow_map_and_gradient = scaler.fit_transform(flow_map_and_gradient.reshape(-1, flow_map_and_gradient.shape[-1])).reshape(flow_map_and_gradient.shape)

            X[i] = flow_map_and_gradient
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_flow_map_and_gradient_2(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            # load mat file contents
            mat_contents = sio.loadmat(mat_file)
            flow_map_mat = mat_contents['Flow_Map']
            lcs_mask = mat_contents['lcs_mask']
            grad_f11 = mat_contents['grad_Flow_Map11']
            grad_f12 = mat_contents['grad_Flow_Map12']
            grad_f21 = mat_contents['grad_Flow_Map21']
            grad_f22 = mat_contents['grad_Flow_Map22']

            # load flow map
            flow_map_shape = np.shape(flow_map_mat)
            flow_map = np.empty([flow_map_shape[1], flow_map_shape[2], flow_map_shape[0] * flow_map_shape[3]], dtype='float32')
            flow_map[:,:,0:2] = flow_map_mat[0,:,:,:]
            flow_map[:,:,2:4] = flow_map_mat[1,:,:,:]
            diff_map = flow_map[:,:,2:4] - flow_map[:,:,0:2]

            # load flow gradient
            grad_shape = [512, 512, 4]
            grad_f = np.empty([grad_shape[0], grad_shape[1], grad_shape[2]], dtype='float32')
            grad_f[:,:,0] = grad_f11
            grad_f[:,:,1] = grad_f21
            grad_f[:,:,2] = grad_f12
            grad_f[:,:,3] = grad_f22

            # load binary mask
            lcs_mask = mat_contents['lcs_mask']

            

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                grad_f = translate_data(grad_f, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                # normalize diff map
                diff_map_shape = np.shape(diff_map)
                diff_map = diff_map.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                max_norm = np.sqrt(np.amax(max_norm))

                diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm

                # normalize grad
                #grad_f_shape = np.shape(grad_f)
                #grad_f = grad_f.reshape(-1, 2)

                #max_norm = np.einsum('ij,ij->i', grad_f, grad_f)
                #max_norm = np.sqrt(np.amax(max_norm))

                #grad_f = grad_f.reshape(grad_f_shape)
                grad_f = grad_f / max_norm

            diff_map_and_gradient = np.concatenate((diff_map, grad_f), axis=2)

            X[i] = diff_map_and_gradient
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_flow_map_gradient(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            
            grad_f11 = mat_contents['grad_Flow_Map11']
            grad_f12 = mat_contents['grad_Flow_Map12']
            grad_f21 = mat_contents['grad_Flow_Map21']
            grad_f22 = mat_contents['grad_Flow_Map22']


            shape = [512, 512, 4]
            grad_f = np.empty([shape[0], shape[1], shape[2]], dtype='float32')
            
            # f11 f12
            # f21 f22

            grad_f[:,:,0] = grad_f11
            grad_f[:,:,1] = grad_f21
            grad_f[:,:,2] = grad_f12
            grad_f[:,:,3] = grad_f22

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                grad_f = translate_data(grad_f, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                grad_f_shape = np.shape(grad_f)
                grad_f = grad_f.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', grad_f, grad_f)
                max_norm = np.sqrt(np.amax(max_norm))

                grad_f = grad_f.reshape(grad_f_shape)
                grad_f = grad_f / max_norm


            X[i] = grad_f
            Y[i] = lcs_mask


        return X, Y

    def __data_generation_velocity_grid_and_flow_map(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            flow_field_u = mat_contents['flow_field_u']
            flow_field_v = mat_contents['flow_field_v']
            flow_shape = np.shape(flow_field_u)
            
            flow_field = np.empty([self.velocity_channels, flow_shape[1], flow_shape[2]], dtype='float32')
            for j in range((self.velocity_channels) // 2):
                flow_field[j * 2] = flow_field_u[j]
                flow_field[2 * j + 1] = flow_field_v[j]
            
            flow_field = np.transpose(flow_field, (1, 2, 0))

            trajectory_mat = mat_contents['Trajectory']
            fm_shape = np.shape(trajectory_mat)
            flow_map = np.empty([fm_shape[0], fm_shape[1], self.n_channels - self.velocity_channels + 2], dtype='float32')

            take = np.round(np.linspace(0, fm_shape[2] // 2 - 1, (self.n_channels-self.velocity_channels) // 2 + 1)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            diff_map = np.empty([fm_shape[0], fm_shape[1], self.n_channels - self.velocity_channels], dtype='float32')
            for idx in range((self.n_channels - self.velocity_channels) // 2):
                diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,idx*2:idx*2+2]
                #diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,0:2]

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis

                flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:
                flow_field_shape = np.shape(flow_field)
                flow_field = flow_field.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_field = flow_field.reshape(flow_field_shape)
                flow_field = flow_field / max_norm

                #diff_map_shape = np.shape(diff_map)
                #diff_map = diff_map.reshape(-1, 2)

                #max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                #max_norm = np.sqrt(np.amax(max_norm))

                #diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm

            flow_field_and_diff_map = np.concatenate((flow_field, diff_map), axis=2)

            X[i] = flow_field_and_diff_map
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_velocity_field_and_gradient(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            flow_field_u = mat_contents['flow_field_u']
            flow_field_v = mat_contents['flow_field_v']

            flow_shape = np.shape(flow_field_u)
            flow_field = np.empty([self.n_channels - 4, flow_shape[1], flow_shape[2]], dtype='float32')
            for j in range((self.n_channels - 4) // 2):
                flow_field[j * 2] = flow_field_u[j]
                flow_field[2 * j + 1] = flow_field_v[j]
            flow_field = np.transpose(flow_field, (1, 2, 0))

            grad_f11 = mat_contents['grad_Flow_Map11']
            grad_f12 = mat_contents['grad_Flow_Map12']
            grad_f21 = mat_contents['grad_Flow_Map21']
            grad_f22 = mat_contents['grad_Flow_Map22']

            grad_f = np.empty([512, 512, 4], dtype='float32')
            grad_f[:,:,0] = grad_f11
            grad_f[:,:,1] = grad_f21
            grad_f[:,:,2] = grad_f12
            grad_f[:,:,3] = grad_f22

            lcs_mask = mat_contents['lcs_mask']


            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
                grad_f = translate_data(grad_f, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:
                flow_field_shape = np.shape(flow_field)
                flow_field = flow_field.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_field = flow_field.reshape(flow_field_shape)
                flow_field = flow_field / max_norm

                grad_f = grad_f / max_norm

            flow_field_and_gradient = np.concatenate((flow_field, grad_f), axis=2)
            X[i] = flow_field_and_gradient
            Y[i] = lcs_mask


        return X, Y

    def __data_generation_cg_tensor(self, file_list_temp):
    
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)

            eigen_vec = mat_contents['cgEigenvector']
            eigen_val = mat_contents['cgEigenvalue']

            eigen_vec = eigen_vec.reshape(*self.dim, 4)
            eigen_val = eigen_val.reshape(*self.dim, 2)

            eigen_vec[:,:,0] = eigen_vec[:,:,0] * eigen_val[:,:,0]
            eigen_vec[:,:,1] = eigen_vec[:,:,1] * eigen_val[:,:,0]
            eigen_vec[:,:,2] = eigen_vec[:,:,2] * eigen_val[:,:,1]
            eigen_vec[:,:,3] = eigen_vec[:,:,3] * eigen_val[:,:,1]
            lcs_mask = mat_contents['lcs_mask']

            # shape flow_field = (512, 512, 12) --> [0, 1] = [x, y]
            # shape lcs_mask = (512, 512) --> [0, 1] = [x, y]
            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                eigen_vec = translate_data(eigen_vec, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                eigen_vec_shape = np.shape(eigen_vec)
                eigen_vec = eigen_vec.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', eigen_vec, eigen_vec)
                max_norm = np.sqrt(np.amax(max_norm))

                eigen_vec = eigen_vec.reshape(eigen_vec_shape)
                eigen_vec = eigen_vec / max_norm
            X[i] = eigen_vec
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_velocity_field_time_sampled(self, file_list_temp): # 3min per epoch wtf
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)


        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)

            flow_field_u = mat_contents['flow_field_u_time_sample_15']
            flow_field_v = mat_contents['flow_field_v_time_sample_15']
            shape = np.shape(flow_field_u)
            
            # zb [10,512, 512]
            flow_field = np.empty([self.n_channels, shape[1], shape[2]], dtype='float32')
            
            take = np.round(np.linspace(0, shape[0] - 1, self.n_channels // 2)).astype(int)
            for idx, val in enumerate(take):
                flow_field[idx*2] = flow_field_u[val]
                flow_field[idx*2+1] = flow_field_v[val]
            
            flow_field = np.transpose(flow_field, (1, 2, 0))
            lcs_mask = mat_contents['lcs_mask']

            # shape flow_field = (512, 512, 12) --> [0, 1] = [x, y]
            # shape lcs_mask = (512, 512) --> [0, 1] = [x, y]
            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                flow_field_shape = np.shape(flow_field)
                flow_field = flow_field.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_field = flow_field.reshape(flow_field_shape)
                flow_field = flow_field / max_norm
            X[i] = flow_field
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_flow_map_2_step(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            
            trajectory_mat = mat_contents['Trajectory']
            shape = np.shape(trajectory_mat)
            flow_map = np.empty([shape[0], shape[1], self.n_channels + 2], dtype='float32')

            take = np.round(np.linspace(0, shape[2] // 2 - 1, self.n_channels // 2 + 1)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            diff_map = np.empty([shape[0], shape[1], self.n_channels], dtype='float32')
            for idx in range(self.n_channels // 2):
                diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,idx*2:idx*2+2]

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:

                diff_map_shape = np.shape(diff_map)
                diff_map = diff_map.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                max_norm = np.sqrt(np.amax(max_norm))

                diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm
            
            X[i] = diff_map
            Y[i] = lcs_mask

        return X, Y

    def __data_generation_velocity_field_time_sampled_and_flow_map(self, file_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=self.dtype)
        Y = np.empty((self.batch_size, *self.dim), dtype= self.dtype)

        # Generate data
        for i, mat_file in enumerate(file_list_temp):
            
            mat_contents = sio.loadmat(mat_file)
            flow_field_u = mat_contents['flow_field_u']
            flow_field_v = mat_contents['flow_field_v']
            flow_shape = np.shape(flow_field_u)
            
            flow_field = np.empty([self.velocity_channels, flow_shape[1], flow_shape[2]], dtype='float32')
            take = np.round(np.linspace(0, flow_shape[0] - 1, self.velocity_channels // 2)).astype(int)
            for idx, val in enumerate(take):
                flow_field[idx*2] = flow_field_u[val]
                flow_field[idx*2+1] = flow_field_v[val]
            
            flow_field = np.transpose(flow_field, (1, 2, 0))

            trajectory_mat = mat_contents['Trajectory']
            fm_shape = np.shape(trajectory_mat)
            flow_map = np.empty([fm_shape[0], fm_shape[1], self.n_channels - self.velocity_channels + 2], dtype='float32')

            take = np.round(np.linspace(0, fm_shape[2] // 2 - 1, (self.n_channels-self.velocity_channels) // 2 + 1)).astype(int)
            for idx, val in enumerate(take):
                flow_map[:,:,idx*2] = trajectory_mat[:,:,val*2]
                flow_map[:,:,idx*2+1] = trajectory_mat[:,:,val*2+1]

            diff_map = np.empty([fm_shape[0], fm_shape[1], self.n_channels - self.velocity_channels], dtype='float32')
            for idx in range((self.n_channels - self.velocity_channels) // 2):
                diff_map[:,:,idx*2:idx*2+2] = flow_map[:,:,2*(idx+1):2*(idx+1)+2] - flow_map[:,:,idx*2:idx*2+2]

            lcs_mask = mat_contents['lcs_mask']

            if self.translate:
                shift_vector = [random.randint(0, 300), random.randint(0, 300)]
                axis_vector = [0, 1] # moving the axis

                flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
                diff_map = translate_data(diff_map, shift_vector=shift_vector, axis_vector=axis_vector)
                lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            
            if self.batch_normalize:
                flow_field_shape = np.shape(flow_field)
                flow_field = flow_field.reshape(-1, 2)

                max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
                max_norm = np.sqrt(np.amax(max_norm))

                flow_field = flow_field.reshape(flow_field_shape)
                flow_field = flow_field / max_norm

                #diff_map_shape = np.shape(diff_map)
                #diff_map = diff_map.reshape(-1, 2)

                #max_norm = np.einsum('ij,ij->i', diff_map, diff_map)
                #max_norm = np.sqrt(np.amax(max_norm))

                #diff_map = diff_map.reshape(diff_map_shape)
                diff_map = diff_map / max_norm

            flow_field_and_diff_map = np.concatenate((flow_field, diff_map), axis=2)

            X[i] = flow_field_and_diff_map
            Y[i] = lcs_mask

        return X, Y

    def get_item(self, file_list_temp):

        if self.input_type == 'Velocity_Field':
            return self.__data_generation_velocity_field(file_list_temp)
        elif self.input_type == 'Flow_Map':
            return self.__data_generation_flow_map(file_list_temp)
        elif self.input_type == 'Flow_Map_Gradient':
            return self.__data_generation_flow_map_gradient(file_list_temp)
        elif self.input_type == 'Flow_Map_and_Gradient':
            return self.__data_generation_flow_map_and_gradient(file_list_temp)
        elif self.input_type == 'Velocity_Field_and_Gradient':
            return self.__data_generation_velocity_field_and_gradient(file_list_temp)
        elif self.input_type == 'Flow_Map_2':
            return self.__data_generation_flow_map_2(file_list_temp)
        elif self.input_type == 'Flow_Map_and_Gradient_2':
            return self.__data_generation_flow_map_and_gradient_2(file_list_temp)
        elif self.input_type == 'Velocity_Field_and_Flow_Map_2':
            return self.__data_generation_velocity_grid_and_flow_map(file_list_temp)
        elif self.input_type == 'CG_Tensor':
            return self.__data_generation_cg_tensor(file_list_temp)
        elif self.input_type == 'Reverse_Flow_Map_2':
            return self.__data_generation_reverse_flow_map_2(file_list_temp)
        elif self.input_type == 'Velocity_Field_TimeSampled':
            return self.__data_generation_velocity_field_time_sampled(file_list_temp)
        elif self.input_type == 'Flow_Map_2_step':
            return self.__data_generation_flow_map_2_step(file_list_temp)
        elif self.input_type == 'Velocity_Field_TimeSampled_and_Flow_Map_2':
            return self.__data_generation_velocity_field_time_sampled_and_flow_map(file_list_temp)

    def translate_data(self, X, shift_vector, axis_vector):
        return np.roll(X, shift_vector, axis=axis_vector)
    
# directly load specified number of mat files at the path location
def load(path, count, shuffle=True, translate= True, normalize=True):

    X = np.empty([count, 512, 512, 12], dtype='float32')
    Y = np.empty([count, 512, 512], dtype='float32')
    
    # All files ending with .mat
    full_dataset = glob.glob(path + "/*.mat")

    if shuffle:
        random.shuffle(full_dataset)

    full_dataset = full_dataset[0:count]
    
    for idx, mat_file in enumerate(full_dataset):
        
        mat_contents = sio.loadmat(mat_file)
        print(mat_file)

        flow_field_u = mat_contents['flow_field_u']
        flow_field_v = mat_contents['flow_field_v']

        shape = np.shape(flow_field_u)
        flow_field = np.empty([shape[0] * 2, shape[1], shape[2]], dtype='float32')
        for i in range(shape[0]):
            flow_field[i * 2] = flow_field_u[i]
            flow_field[2 * i + 1] = flow_field_v[i]

        flow_field = np.transpose(flow_field, (1, 2, 0))
        lcs_mask = mat_contents['lcs_mask']

        # shape flow_field = (512, 512, 12) --> [0, 1] = [x, y]
        # shape lcs_mask = (512, 512) --> [0, 1] = [x, y]
        if translate:
            
            shift_vector = [random.randint(0, 250), random.randint(0, 250)]
            axis_vector = [0, 1] 
            flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
            lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
            # shape lcs mask = ()

        if normalize:
            scaler = StandardScaler()
            flow_field = scaler.fit_transform(flow_field.reshape(-1, flow_field.shape[-1])).reshape(flow_field.shape)


        X[idx] = flow_field
        Y[idx] = lcs_mask

    return X, Y

# directly load specified number of mat files at the path location
def load_all(path, count, shape, channels, shuffle=True, translate= True, normalize=True):

    all_flow_fields = np.empty([count, *shape, channels], dtype='float32')
    all_flow_maps = np.empty([count, *shape, 4], dtype='float32')
    all_flow_map_grads = np.empty([count, *shape, 4], dtype='float32')
    all_lcs_masks = np.empty([count, *shape], dtype='float32')
    
    # All files ending with .mat
    if type(path) == str:
        full_dataset = glob.glob(path + "/*.mat")
    else:
        full_dataset = path

    if shuffle:
        random.shuffle(full_dataset)

    full_dataset = full_dataset[0:count]
    
    for idx, mat_file in enumerate(full_dataset):
        
        mat_contents = sio.loadmat(mat_file)

        flow_field_u = mat_contents['flow_field_u']
        flow_field_v = mat_contents['flow_field_v']
        flow_map = mat_contents['Flow_Map']
        grad_f11 = mat_contents['grad_Flow_Map11']
        grad_f12 = mat_contents['grad_Flow_Map12']
        grad_f21 = mat_contents['grad_Flow_Map21']
        grad_f22 = mat_contents['grad_Flow_Map22']


        flow_field = np.empty([channels, shape[0], shape[1]], dtype='float32')
        for j in range(channels // 2):
            flow_field[j * 2] = flow_field_u[j]
            flow_field[2 * j + 1] = flow_field_v[j]

        flow_field = np.transpose(flow_field, (1, 2, 0))
        

        grad_shape = [512, 512, 4]
        grad_f = np.empty([grad_shape[0], grad_shape[1], grad_shape[2]], dtype='float32')

        grad_f[:,:,0] = grad_f11
        grad_f[:,:,1] = grad_f12
        grad_f[:,:,2] = grad_f21            
        grad_f[:,:,3] = grad_f22

        lcs_mask = mat_contents['lcs_mask']


        if translate:
            
            shift_vector = [random.randint(0, 250), random.randint(0, 250)]
            axis_vector = [0, 1]
            flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
            flow_map = translate_data(flow_map, shift_vector=shift_vector, axis_vector=axis_vector)
            grad_f = translate_data(grad_f, shift_vector=shift_vector, axis_vector=axis_vector)
            lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)


        if normalize:          
            flow_field_shape = np.shape(flow_field)
            flow_field = flow_field.reshape(-1, 2)
        
            max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
            max_norm = np.sqrt(np.amax(max_norm))

            flow_field = flow_field.reshape(flow_field_shape)
            flow_field = flow_field / max_norm


        all_flow_fields[idx] = flow_field
        all_flow_maps[idx] = flow_map
        all_flow_map_grads[idx] = grad_f
        all_lcs_masks[idx] = lcs_mask

    return all_flow_fields, all_flow_maps, all_flow_map_grads, all_lcs_masks, full_dataset

def load_specific(path, shape, channels, translate= True, normalize=True):

    all_flow_fields = np.empty([1,*shape, channels], dtype='float32')
    all_flow_maps = np.empty([1,*shape, 4], dtype='float32')
    all_flow_map_grads = np.empty([1,*shape, 4], dtype='float32')
    all_lcs_masks = np.empty([1,*shape], dtype='float32')
    all_flow_fields_sampletime = np.empty([1,*shape, channels], dtype='float32')
    
    # All files ending with .mat
    full_dataset = glob.glob(path + "/*.mat")

    mat_contents = sio.loadmat(path)

    flow_field_u = mat_contents['flow_field_u']
    flow_field_v = mat_contents['flow_field_v']
    flow_map = mat_contents['Flow_Map']
    grad_f11 = mat_contents['grad_Flow_Map11']
    grad_f12 = mat_contents['grad_Flow_Map12']
    grad_f21 = mat_contents['grad_Flow_Map21']
    grad_f22 = mat_contents['grad_Flow_Map22']

    flow_field_u_mat = mat_contents['flow_field_u_time_sample_15']
    flow_field_v_mat = mat_contents['flow_field_v_time_sample_15']

    flow_field = np.empty([channels, shape[0], shape[1]], dtype='float32')
    for j in range(channels // 2):
        flow_field[j * 2] = flow_field_u[j]
        flow_field[2 * j + 1] = flow_field_v[j]
    flow_field = np.transpose(flow_field, (1, 2, 0))
    
    flow_field_sampletime = np.empty([channels, shape[1], shape[2]], dtype='float32') 
    take = np.round(np.linspace(0, shape[0] - 1, channels // 2)).astype(int)
    for idx, val in enumerate(take):
        flow_field_sampletime[idx*2] = flow_field_u_mat[val]
        flow_field_sampletime[idx*2+1] = flow_field_v_mat[val]
    flow_field_sampletime = np.transpose(flow_field_sampletime, (1, 2, 0))

    lcs_mask = mat_contents['lcs_mask']

    grad_shape = [512, 512, 4]
    grad_f = np.empty([grad_shape[0], grad_shape[1], grad_shape[2]], dtype='float32')

    grad_f[:,:,0] = grad_f11
    grad_f[:,:,1] = grad_f12
    grad_f[:,:,2] = grad_f21            
    grad_f[:,:,3] = grad_f22

    lcs_mask = mat_contents['lcs_mask']

    if translate:
            
        shift_vector = [random.randint(0, 250), random.randint(0, 250)]
        axis_vector = [0, 1]
        flow_field = translate_data(flow_field, shift_vector=shift_vector, axis_vector=axis_vector)
        flow_map = translate_data(flow_map, shift_vector=shift_vector, axis_vector=axis_vector)
        grad_f = translate_data(grad_f, shift_vector=shift_vector, axis_vector=axis_vector)
        lcs_mask = translate_data(lcs_mask, shift_vector=shift_vector, axis_vector=axis_vector)
        flow_field_sampletime = translate_data(flow_field_sampletime, shift_vector=shift_vector, axis_vector=axis_vector)



    if normalize:          
        flow_field_shape = np.shape(flow_field)
        flow_field = flow_field.reshape(-1, 2)
        
        max_norm = np.einsum('ij,ij->i', flow_field, flow_field)
        max_norm = np.sqrt(np.amax(max_norm))

        flow_field = flow_field.reshape(flow_field_shape)
        flow_field = flow_field / max_norm

        flow_field_sampletime_shape = np.shape(flow_field_sampletime)
        flow_field_sampletime = flow_field_sampletime.reshape(-1, 2)
        
        max_norm = np.einsum('ij,ij->i', flow_field_sampletime, flow_field_sampletime)
        max_norm = np.sqrt(np.amax(max_norm))

        flow_field_sampletime = flow_field_sampletime.reshape(flow_field_sampletime_shape)
        flow_field_sampletime = flow_field_sampletime / max_norm


    all_flow_fields[0] = flow_field
    all_flow_maps[0] = flow_map
    all_flow_map_grads[0] = grad_f
    all_lcs_masks[0] = lcs_mask
    all_flow_fields_sampletime[0] = flow_field_sampletime

    return all_flow_fields, all_flow_maps, all_flow_map_grads, all_lcs_masks, all_flow_fields_sampletime, full_dataset

def translate_data(X, shift_vector, axis_vector):
    return np.roll(X, shift_vector, axis=axis_vector)

def translate_test1():
    array = np.arange(24).reshape(2, 3, 4)
    print("Original array : \n", array)

    print("\nRolling axis 0 with shift 1: \n", translate_data(array, [1], [0]))

    print("\nRolling axis 1 with shift 2: \n", translate_data(array, [1], [1]))

    print("\nRolling axis 2 with shift 2: \n", translate_data(array, [2], [2]))

    print("\nRolling axis 1,2 with shift 1,2: \n", translate_data(array, [1,2], [1,2]))
    
def translate_test2():
    result_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results10 clean true'
    X, Y = load(path=result_path, count=1, shuffle=True, translate=False)

    # make grid
    x = np.linspace(start=0, stop=1, num=512)
    y = np.linspace(start=0, stop=1, num=512)
    x, y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(2,2)
    ax[0, 0].streamplot(x, y, X[0,:,:,0], X[0,:,:,1], density = 0.5)
    ax[0, 0].set_title('Flow Field before translation')
    ax[1, 0].imshow(Y[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[1, 0].set_title('Mask,before translation')

    shift_vector = [0, 250]
    axis_vector = [1, 2] 
    X = translate_data(X, shift_vector=shift_vector, axis_vector=axis_vector)
    Y = translate_data(Y, shift_vector=shift_vector, axis_vector=axis_vector)

    ax[0, 1].streamplot(x, y, X[0,:,:,0], X[0,:,:,1], density = 0.5)
    ax[0, 1].set_title('Flow Field after translation')
    ax[1, 1].imshow(Y[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[1, 1].set_title('Mask after translation')

    plt.show()
    print('Test finished!')

def test_normalization():
    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean'
    flow_field, flow_map, flow_map_grad, lcs_mask, full_dataset = load_all(test_data_path, count=1, translate=False, normalize=False)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    x, y = np.meshgrid(X, Y)
    
    step = 25
    fig, ax = plt.subplots(2,3)
    #ax[0, 0].streamplot(x, y, flow_field[0,:,:,0], flow_field[0,:,:,1], density = 0.5)
    #ax[0, 0].set_title('Flow Field')
    #ax[0, 0].streamplot(x[::step,::step], y[::step,::step], flow_field[0,::step,::step,0], flow_field[0,::step,::step,1], 1)
    #ax[0, 0].set_title('Flow Field')
    ax[0, 1].imshow(lcs_mask[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    ax[0, 1].set_title('Mask')
    ax[0, 2].quiver(flow_map[0,::step,::step,0], flow_map[0,::step,::step,1], flow_map[0,::step,::step,2], flow_map[0,::step,::step,3])
    ax[0, 2].set_title('Flow Map')

    u_interp = RegularGridInterpolator((X,Y,Z), flow_field[0,:,:,0::2], bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((X,Y,Z), flow_field[0,:,:,1::2], bounds_error=False, fill_value=None)

    t = np.linspace(0, 0.6, 25)
    for ypos in Y[::100]:
        for xpos in X[::100]:
            sol = calc_pathline(u_interp, v_interp, xpos, ypos, t)
            ax[0,0].plot(sol[:, 0], sol[:, 1])

    scaler = StandardScaler()
    flow_field_shape = flow_field.shape
    flow_field_1 = flow_field.reshape(-1, flow_field_shape[-1])
    flow_field_1 = scaler.fit_transform(flow_field_1)
    flow_field_1 = flow_field_1.reshape(flow_field_shape)

    scaler = StandardScaler()
    flow_field_shape = flow_field.shape
    flow_field_2 = flow_field.reshape(-1, flow_field_shape[-1])
    flow_field_2[:,0::2] = scaler.fit_transform(flow_field_2[:,0::2])
    flow_field_2[:,1::2] = scaler.fit_transform(flow_field_2[:,1::2])
    flow_field_2 = flow_field_2.reshape(flow_field_shape)

    scaler = StandardScaler()
    flow_map_shape = flow_map.shape
    flow_map_1 = flow_map.reshape(-1, flow_map_shape[-1])
    flow_map_1 = scaler.fit_transform(flow_map_1)
    flow_map_1 = flow_map_1.reshape(flow_map_shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    flow_map_shape = flow_map.shape
    flow_map_2 = flow_map.reshape(-1, flow_map_shape[-1])
    flow_map_2 = scaler.fit_transform(flow_map_2)
    flow_map_2 = flow_map_2.reshape(flow_map_shape)

    #scaler = StandardScaler()
    #grad_f = scaler.fit_transform(grad_f.reshape(-1, grad_f.shape[-1])).reshape(grad_f.shape)

    ax[1, 0].streamplot(x[::step,::step], y[::step,::step], flow_field_1[0,::step,::step,0], flow_field_1[0,::step,::step,1], 1)
    ax[1, 0].set_title('Flow Field')

    ax[1, 1].streamplot(x[::step,::step], y[::step,::step], flow_field_2[0,::step,::step,0], flow_field_2[0,::step,::step,1], 1)
    ax[1, 1].set_title('Flow Field')

    ax[1, 2].streamplot(x[::step,::step], y[::step,::step], flow_field_1[0,::step,::step,0] - flow_field_2[0,::step,::step,0], flow_field_1[0,::step,::step,1] - flow_field_2[0,::step,::step,0], 1)
    ax[1, 2].set_title('Flow Field')

    #ax[1, 1].quiver(x[::step,::step], y[::step,::step], flow_field_1[0,::step,::step,0], flow_field_1[0,::step,::step,1], 1)
    #ax[1, 1].set_title('Flow Field')

    #ax[1, 1].quiver(flow_map_1[0,::step,::step,0], flow_map_1[0,::step,::step,1], flow_map_1[0,::step,::step,2], flow_map_1[0,::step,::step,3], 1)
    #ax[1, 1].set_title('Flow Map')

    #ax[1, 2].quiver(flow_map_2[0,::step,::step,0], flow_map_2[0,::step,::step,1], flow_map_2[0,::step,::step,2], flow_map_2[0,::step,::step,3], 1)
    #ax[1, 2].set_title('Flow Map')


    #ax[1, 1].imshow(lcs_mask[0,:,:], interpolation='bilinear', origin='lower', cmap=cm.Greys)
    #ax[1, 1].set_title('Mask')
    #ax[1, 2].quiver(flow_map[0,::step,::step,0], flow_map[0,::step,::step,1], flow_map[0,::step,::step,2], flow_map[0,::step,::step,3])
    #ax[1, 2].set_title('Flow Map')

    plt.show()
    print('Test finished!')

def test_normalization_velocity():
    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean'
    flow_field, flow_map, flow_map_grad, lcs_mask, full_dataset = load_all(test_data_path, count=1, translate=False, normalize=False)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    x, y = np.meshgrid(X, Y)
    
    step = 25
    density = 0.5
    fig, ax = plt.subplots(2,3)
    ax[0, 0].streamplot(x, y, flow_field[0,:,:,0], flow_field[0,:,:,1], density)
    ax[0, 0].set_title('Flow Field')

    u_interp = RegularGridInterpolator((X,Y,Z), flow_field[0,:,:,0::2], bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((X,Y,Z), flow_field[0,:,:,1::2], bounds_error=False, fill_value=None)
    t = np.linspace(0, 0.6, 5)
    for ypos in Y[::25]:
        for xpos in X[::25]:
            sol = calc_pathline(u_interp, v_interp, xpos, ypos, t)
            ax[1,0].plot(sol[:, 0], sol[:, 1])

    flow_field_shape = flow_field.shape
    scaler = StandardScaler()
    flow_field_1 = flow_field.copy()
    flow_field_1 = flow_field_1.reshape(-1, flow_field_shape[-1])
    flow_field_1 = scaler.fit_transform(flow_field_1)
    flow_field_1 = flow_field_1.reshape(flow_field_shape)

    scaler = StandardScaler()
    flow_field_2 = flow_field.copy()
    flow_field_2 = flow_field_2.reshape(-1, flow_field_shape[-1])
    flow_field_2[:,0::2] = scaler.fit_transform(flow_field_2[:,0::2])
    flow_field_2[:,1::2] = scaler.fit_transform(flow_field_2[:,1::2])
    flow_field_2 = flow_field_2.reshape(flow_field_shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    flow_field_3 = flow_field.copy()
    flow_field_3 = flow_field_3.reshape(-1, flow_field_shape[-1])
    flow_field_3 = scaler.fit_transform(flow_field_3)
    flow_field_3 = flow_field_3.reshape(flow_field_shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    flow_field_4 = flow_field.copy()
    flow_field_4 = flow_field_4.reshape(-1, flow_field_shape[-1])
    flow_field_4[:,0::2] = scaler.fit_transform(flow_field_4[:,0::2])
    flow_field_4[:,1::2] = scaler.fit_transform(flow_field_4[:,1::2])
    flow_field_4 = flow_field_4.reshape(flow_field_shape)


    ax[0, 1].streamplot(x[::step,::step], y[::step,::step], flow_field_1[0,::step,::step,0], flow_field_1[0,::step,::step,1], density)
    ax[0, 1].set_title('u,v standard togheter')

    ax[0, 2].streamplot(x[::step,::step], y[::step,::step], flow_field_2[0,::step,::step,0], flow_field_2[0,::step,::step,1], density)
    ax[0, 2].set_title('u,v standard seperatly')

    ax[1, 1].streamplot(x[::step,::step], y[::step,::step], flow_field_3[0,::step,::step,0], flow_field_3[0,::step,::step,1], density)
    ax[1, 1].set_title('u,v minmax togheter')

    ax[1, 2].streamplot(x[::step,::step], y[::step,::step], flow_field_4[0,::step,::step,0], flow_field_4[0,::step,::step,1], density)
    ax[1, 2].set_title('u,v minmax seperatly')

    plt.show()
    print('Test finished!')

def test_normalization_flow_map():
    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set_new'
    _, flow_map, _, _, _ = load_all(test_data_path, count=1, translate=False, normalize=False, shape=(512,512), channels=4)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    x, y = np.meshgrid(X, Y)
    step = 10
    density = 3

    fig, ax = plt.subplots(2,2)
    flow_map = flow_map[0]
    flow_map_copy = flow_map.copy()
    diff_map_copy = flow_map[:,:,2:4] - flow_map[:,:,0:2]
    diff_map = flow_map[:,:,2:4] - flow_map[:,:,0:2]
    ax[0, 0].streamplot(flow_map[::step,::step,0], flow_map[::step,::step,1], diff_map[::step,::step,0], diff_map[::step,::step,1], density)
    ax[0, 0].set_title('Flow Map')

    flow_map_shape = np.shape(flow_map)
    flow_map = flow_map.reshape(-1, 2)     
    max_norm = np.einsum('ij,ij->i', flow_map, flow_map)
    max_norm = np.sqrt(np.amax(max_norm))
    flow_map = flow_map.reshape(flow_map_shape)
    flow_map = flow_map / max_norm

    diff_map_shape = np.shape(diff_map_copy)
    diff_map_copy = diff_map_copy.reshape(-1, 2)     
    max_norm = np.einsum('ij,ij->i', diff_map_copy, diff_map_copy)
    max_norm = np.sqrt(np.amax(max_norm))
    diff_map_copy = diff_map_copy.reshape(diff_map_shape)
    diff_map_copy = diff_map_copy / max_norm
    
    diff_map = flow_map[:,:,2:4] - flow_map[:,:,0:2]
    ax[1, 0].streamplot(flow_map[::step,::step,0], flow_map[::step,::step,1], diff_map[::step,::step,0], diff_map[::step,::step,1], density)
    ax[1, 0].set_title('Flow Map norm')

    ax[1, 1].streamplot(flow_map[::step,::step,0], flow_map[::step,::step,1], diff_map_copy[::step,::step,0], diff_map_copy[::step,::step,1], density)
    ax[1, 1].set_title('Flow Map norm')

    test = diff_map - diff_map_copy
    plt.show()
    print('Test finished!')

def test_normalization_flow_map_derivative():
    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean'
    flow_field, flow_map, flow_map_grad, lcs_mask, full_dataset = load_all(test_data_path, count=1, translate=False, normalize=False)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    x, y = np.meshgrid(X, Y)
    step = 15
    density = 1

    fig, ax = plt.subplots(2,4)

    ax[0, 0].streamplot(x[::step,::step], y[::step,::step], flow_map_grad[0,::step,::step,0], flow_map_grad[0,::step,::step,2], density)
    ax[0, 0].set_title('Flow Map Derivative xdx ydx')
    
    ax[1, 0].streamplot(x[::step,::step], y[::step,::step], flow_map_grad[0,::step,::step,1], flow_map_grad[0,::step,::step,3], density)
    ax[1, 0].set_title('Flow Map Derivative xdy ydy')


    flow_map_grad_shape = flow_map_grad.shape

    scaler = StandardScaler()
    flow_map_grad_1 = flow_map_grad.copy()
    flow_map_grad_1 = flow_map_grad_1.reshape(-1, flow_map_grad_shape[-1])
    flow_map_grad_1 = scaler.fit_transform(flow_map_grad_1)
    flow_map_grad_1 = flow_map_grad_1.reshape(flow_map_grad_shape)

    scaler = StandardScaler()
    flow_map_grad_2 = flow_map_grad.copy()
    flow_map_grad_2 = flow_map_grad_2.reshape(-1, flow_map_grad_shape[-1])
    flow_map_grad_2[:,0:4:2] = scaler.fit_transform(flow_map_grad_2[:,0:4:2])
    flow_map_grad_2[:,1:4:2] = scaler.fit_transform(flow_map_grad_2[:,1:4:2])
    flow_map_grad_2 = flow_map_grad_2.reshape(flow_map_grad_shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_map_grad_3 = flow_map_grad.copy()
    flow_map_grad_3 = flow_map_grad_3.reshape(-1, flow_map_grad_shape[-1])
    flow_map_grad_3 = scaler.fit_transform(flow_map_grad_3)
    flow_map_grad_3 = flow_map_grad_3.reshape(flow_map_grad_shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    flow_map_grad_4 = flow_map_grad.copy()
    flow_map_grad_4 = flow_map_grad_4.reshape(-1, flow_map_grad_shape[-1])
    flow_map_grad_4[:,0:4:2] = scaler.fit_transform(flow_map_grad_4[:,0:4:2])
    flow_map_grad_4[:,1:4:2] = scaler.fit_transform(flow_map_grad_4[:,1:4:2])
    flow_map_grad_4 = flow_map_grad_4.reshape(flow_map_grad_shape)

    ax[0, 1].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_1[0,::step,::step,0], flow_map_grad_1[0,::step,::step,2], density)
    ax[0, 1].set_title('DF standard togheter')
    ax[1, 1].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_1[0,::step,::step,1], flow_map_grad_1[0,::step,::step,3], density)
    ax[1, 1].set_title('DF standard togheter')

    ax[0, 2].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_2[0,::step,::step,0], flow_map_grad_2[0,::step,::step,2], density)
    ax[0, 2].set_title('DF standard seperated')
    ax[1, 2].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_2[0,::step,::step,1], flow_map_grad_2[0,::step,::step,3], density)
    ax[1, 2].set_title('DF standar seperated')

    ax[0, 3].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_4[0,::step,::step,0], flow_map_grad_4[0,::step,::step,2], density)
    ax[0, 3].set_title('DF minmax togheter')
    ax[1, 3].streamplot(x[::step, ::step], y[::step, ::step], flow_map_grad_4[0,::step,::step,1], flow_map_grad_4[0,::step,::step,3], density)
    ax[1, 3].set_title('DF minmax togheter')



    plt.show()
    print('Test finished!')

def test_max_norm_normalization():
    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean'
    flow_field, flow_map, flow_map_grad, lcs_mask, full_dataset = load_all(test_data_path, count=1, translate=False, normalize=False)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    x, y = np.meshgrid(X, Y)
    
    step = 25
    density = 1
    fig, ax = plt.subplots(2,4)
    ax[0, 0].quiver(x[::step,::step], y[::step,::step], flow_field[0,::step,::step,0], flow_field[0,::step,::step,1], 1)
    ax[0, 0].set_title('Flow Field')

    diff_map = flow_map[0,:,:,2:4] - flow_map[0,:,:,0:2]
    ax[0, 1].quiver(flow_map[0,::step,::step,0], flow_map[0,::step,::step,1], diff_map[::step,::step,0], diff_map[::step,::step,1], density)
    ax[0, 1].set_title('Flow Map')

    ax[0, 2].quiver(x[::step,::step], y[::step,::step], flow_map_grad[0,::step,::step,0], flow_map_grad[0,::step,::step,2]), density
    ax[0, 2].set_title('Flow Map Derivative')

    ax[0, 3].quiver(x[::step,::step], y[::step,::step], flow_map_grad[0,::step,::step,0], flow_map_grad[0,::step,::step,2]), density
    ax[0, 3].set_title('Flow Map Derivative')

    flow_field = flow_field[0]
    flow_field_shape = np.shape(flow_field)
    flow_field_norm = flow_field.copy()
    flow_field_norm = flow_field_norm.reshape(-1, 2)

    max_norm = np.einsum('ij,ij->i', flow_field_norm, flow_field_norm)
    max_normV = np.sqrt(np.amax(max_norm))

    flow_field_norm = flow_field_norm.reshape(flow_field_shape)
    flow_field_norm = flow_field_norm / max_normV

    #

    diff_map_norm = diff_map.copy()
    diff_map_shape = np.shape(diff_map)
    diff_map_norm = diff_map_norm.reshape(-1, 2)

    max_norm = np.einsum('ij,ij->i', diff_map_norm, diff_map_norm)
    max_norm = np.sqrt(np.amax(max_norm))

    diff_map_norm = diff_map_norm.reshape(diff_map_shape)
    diff_map_norm = diff_map_norm / max_norm

    #
    flow_map_grad = flow_map_grad[0]
    print(np.shape(flow_map_grad))
    flow_map_grad_shape = np.shape(flow_map_grad)
    flow_map_grad_norm = flow_map_grad.copy()
    flow_map_grad_norm = flow_map_grad_norm.reshape(-1, 4)

    max_norm = np.einsum('ij,ij->i', flow_map_grad_norm, flow_map_grad_norm)
    max_norm = np.sqrt(np.amax(max_norm))

    flow_map_grad_norm = flow_map_grad_norm.reshape(flow_map_grad_shape)
    flow_map_grad_norm = flow_map_grad_norm / max_norm

    flow_map_grad_Vnorm = flow_map_grad.copy()
    flow_map_grad_Vnorm = flow_map_grad_Vnorm / (max_normV * 100)

    ax[1, 0].quiver(x[::step,::step], y[::step,::step], flow_field_norm[::step,::step,0], flow_field_norm[::step,::step,1], 1)
    ax[1, 0].set_title('Flow Field')

    ax[1, 1].quiver(flow_map[0,::step,::step,0], flow_map[0,::step,::step,1], diff_map_norm[::step,::step,0], diff_map_norm[::step,::step,1], density)
    ax[1, 1].set_title('Flow Map')

    ax[1, 2].quiver(x[::step,::step], y[::step,::step], flow_map_grad_norm[::step,::step,0], flow_map_grad_norm[::step,::step,2])
    ax[1, 2].set_title('Flow Map Derivative')

    ax[1, 3].quiver(x[::step,::step], y[::step,::step], flow_map_grad_Vnorm[::step,::step,0], flow_map_grad_Vnorm[::step,::step,2])
    ax[1, 3].set_title('Flow Map Derivative')

    plt.show()
    print("wuuh")

def calc_pathline(u_interp, v_interp, xpos, ypos, t):

    y0 = [ypos, xpos]
    sol = odeint(pathline, y0, t, args=(u_interp, v_interp))
    return sol

def pathline(y, t, u, v):

    y = np.mod(y, 1.001)
    t = np.mod(t, 10.00001)

    du = u([y[1], y[0], t])[0]
    dv = v([y[1], y[0], t])[0]
    dydt = [du, dv]
    return dydt

def vector_to_rgb(angle, absolute):
    """Get the rgb value for the given `angle` and the `absolute` value

    Parameters
    ----------
    angle : float
        The angle in radians
    absolute : float
        The absolute value of the gradient
    
    Returns
    -------
    array_like
        The rgb value as a tuple with values [0..1]
    """
    global max_abs

    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi

    return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                         absolute / max_abs, 
                                         absolute / max_abs))

def visualize_flow_map():

    test_data_path = r'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set_new'
    _, flow_map, _, lcs_mask, _ = load_all(test_data_path, count=1, translate=False, normalize=False, shape=(512,512), channels=4)

    # make grid
    X = np.linspace(start=0, stop=1, num=512)
    Y = np.linspace(start=0, stop=1, num=512)
    Z = np.linspace(0,0.6,6)
    X, Y = np.meshgrid(X, Y)
    step = 10
    density = 3

    lcs_mask = lcs_mask[0]
    flow_map = flow_map[0]
    U = flow_map[:,:,1]
    V = flow_map[:,:,2]

    angles = np.arctan2(V, U)
    lengths = np.sqrt(np.square(U) + np.square(V))
    global max_abs
    max_abs = np.max(lengths)
    fig, ax = plt.subplots(2,3)
    # color is direction, hue and value are magnitude
    c1 = np.array(list(map(vector_to_rgb, angles.flatten(), lengths.flatten())))

    ax[0,0].set_title("Color is lenth,\nhue and value are magnitude")
    c1 = c1.reshape([512, 512, 3])
    c1 = c1 + np.array([0.0, 0.0, 0.3])
    q = ax[0,0].imshow(c1, interpolation='bilinear', origin='lower')

    # color is length only
    c2 = np.array(list(map(vector_to_rgb, angles.flatten(), 
                                        np.ones_like(lengths.flatten()) * max_abs)))
    ax[0,1].set_title("Color is direction only")
    c2 = c2.reshape([512, 512, 3])
    q = ax[0,1].imshow(c2, interpolation='bilinear', origin='lower')


    # color is direction only
    c3 = np.array(list(map(vector_to_rgb, 2 * np.pi * lengths.flatten() / max_abs, 
                                        max_abs * np.ones_like(lengths.flatten()))))
    # create one-length vectors
    U_ddash = np.ones_like(U)
    V_ddash = np.zeros_like(V)
    # now rotate them
    U_dash = U_ddash * np.cos(angles) - V_ddash * np.sin(angles)
    V_dash = U_ddash * np.sin(angles) + V_ddash * np.cos(angles)

    ax[0,2].set_title("Uniform length,\nColor is magnitude only")
    c3 = c3.reshape([512, 512, 3])
    q = ax[0,2].imshow(-c3 + c2, interpolation='bilinear', origin='lower')

    q = ax[1,1].imshow(lcs_mask, interpolation='bilinear', origin='lower')
    plt.show()
    print('hi')

#visualize_flow_map()
#test_normalization_velocity()
#test_normalization_flow_map()
#test_normalization_flow_map_derivative()
#test_max_norm_normalization()
