clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Validation_Data_Set';
cd(result_path);
listing = dir(result_path);
sampling_rate = 1;

for i = 3:size(listing, 1)
%% Construct vortex helper
result = listing(i).name;
load(result);
display(pathIn_num_str)

pathIn = ['E:\Fundue\', pathIn_num_str ,'.am'];
vhelp = vortex_helper(pathIn, sampling_rate);
time = str2double(time_str);
interval = str2double(interval_str);
timespan = [time, time+interval];
data_timespan = fix(transform_to_DataCoords(vhelp, timespan, 3));

% define unsteady velocity field
% permute tensor such that time is the first dimension
flow_field = get_flow_vector_field3D(vhelp);
flow_field.u = permute(flow_field.u, [3,1,2]);
flow_field.v = permute(flow_field.v, [3,1,2]);


time_step = 15;
%time_step_count = 25;
%flow_field_u = flow_field.u(data_timespan(1):time_step_size:data_timespan(1) + time_step_count, :, :);
%flow_field_v = flow_field.v(data_timespan(1):time_step_size:data_timespan(1) + time_step_count, :, :);

flow_field_u_time_sample_15 = flow_field.u(data_timespan(1):time_step:data_timespan(2), :, :);
flow_field_v_time_sample_15 = flow_field.v(data_timespan(1):time_step:data_timespan(2), :, :);

save(result, 'flow_field_u_time_sample_15', 'flow_field_v_time_sample_15','-append');
end
