clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Validation_Data_Set';
cd(result_path);
listing = dir(result_path);
sampling_rate = 10;

for i = 3:size(listing, 1)
%% Construct vortex helper
result = listing(i).name;
load(result);

time = str2double(time_str);
interval = str2double(interval_str);
timespan = [time, time+interval];

pathIn = ['E:\Fundue\', pathIn_num_str ,'.am'];
vhelp = vortex_helper(pathIn, sampling_rate);

%% Skip lots of definitions
domain = [vhelp.domainMin(1),vhelp.domainMax(1);vhelp.domainMin(2),vhelp.domainMax(2)];
boundary_epsilon = 0.0002;
eps_domain = domain + [-boundary_epsilon, boundary_epsilon; -boundary_epsilon, boundary_epsilon];

% set main grid resolution
sampled_grid_resolution = get_sampled_resolution(vhelp, 1);
resolutionX = sampled_grid_resolution;
[resolutionY,~] = equal_resolution(domain,resolutionX);
resolution = [resolutionX,resolutionY];

% Velocity field definition
% define grid points for griddedInterpolant
xy = transpose(get_mesh_grid1D(vhelp, 1));
time = transpose(get_mesh_grid1D(vhelp, 3));

% define unsteady velocity field
% permute tensor such that time is the first dimension
flow_field = get_flow_vector_field3D(vhelp);
flow_field.u = permute(flow_field.u, [3,1,2]);
flow_field.v = permute(flow_field.v, [3,1,2]);
periodicBc = [true, true];

% define griddedInterpolant
interpMethod = 'linear';
extrapMethod = 'linear';
u_interpolant = griddedInterpolant({time,xy,xy},flow_field.u,interpMethod,extrapMethod);
v_interpolant = griddedInterpolant({time,xy,xy},flow_field.v,interpMethod,extrapMethod);

% define function lDerivative: extracts velocity at input pos and time
lDerivative = @(t,y,~)derivative(t,y,u_interpolant,v_interpolant, periodicBc, eps_domain);
SolverOptions = odeset('relTol',1e-5,'initialStep',1e-6);

%% Calculate the Trajectory
trajectory_npoints = 25;
Trajectory = get_trajectory(lDerivative,domain,resolution,timespan,trajectory_npoints, SolverOptions);

Flow_Map = Trajectory(:,:,[1,2,trajectory_npoints*2-1,trajectory_npoints*2]); % first and last positions of the trajectory
[grad_Flow_Map11,grad_Flow_Map12,grad_Flow_Map21,grad_Flow_Map22] = get_flowmap_gradient(Flow_Map);

disp('Trajectory Calculated');

save(result, 'Trajectory', 'Flow_Map','grad_Flow_Map11', 'grad_Flow_Map12', 'grad_Flow_Map21', 'grad_Flow_Map22', '-append'); 

end
