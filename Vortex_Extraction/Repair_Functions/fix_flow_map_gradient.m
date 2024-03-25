clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set';
current_path = pwd;
cd(result_path);
listing = dir(result_path);
sampling_rate = 1;

for i = 3:size(listing, 1)
%% Construct vortex helper
result = listing(i).name;
load(result);

[grad_Flow_Map11,grad_Flow_Map12,grad_Flow_Map21,grad_Flow_Map22] = get_flowmap_gradient(Flow_Map);
disp('Flow Map gradient calculated');

save(result,'grad_Flow_Map11', 'grad_Flow_Map12', 'grad_Flow_Map21', 'grad_Flow_Map22', '-append');
 
end

cd(current_path);