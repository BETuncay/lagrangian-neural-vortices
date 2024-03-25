% used for testing various program snippets

clc;
clear;
close all;

sampling_rate = 1;
pathIn_num = 2000;
pathIn_num_str = num2str(pathIn_num,'%04.f');
pathIn = ['E:\Fundue\', pathIn_num_str ,'.am'];
vhelp = vortex_helper(pathIn, sampling_rate);

% define domain
resolution = [512,512];
domain = [vhelp.domainMin(1),vhelp.domainMax(1);vhelp.domainMin(2),vhelp.domainMax(2)];
boundary_epsilon = 0.0002;
eps_domain = domain + [-boundary_epsilon, boundary_epsilon; -boundary_epsilon, boundary_epsilon];
time = 3;
timespan = [3, 3.1];
data_time = fix(transform_to_DataCoords(vhelp, time, 3));

% define grid points for griddedInterpolant
xgrid = transpose(get_mesh_grid1D(vhelp, 1));
ygrid = transpose(get_mesh_grid1D(vhelp, 2));
time = transpose(get_mesh_grid1D(vhelp, 3));

% define unsteady velocity field
% permute tensor such that time is the first dimension
flow_field = get_flow_vector_field3D(vhelp);
flow_field.u = permute(flow_field.u, [3,1,2]);
flow_field.v = permute(flow_field.v, [3,1,2]);

% set velocity field properties
incompressible = true;
periodicBc = [true, true];

% define griddedInterpolant
interpMethod = 'linear';
extrapMethod = 'linear';
u_interpolant = griddedInterpolant({time,ygrid,xgrid},flow_field.u,interpMethod,extrapMethod);
v_interpolant = griddedInterpolant({time,ygrid,xgrid},flow_field.v,interpMethod,extrapMethod);

% define function lDerivative: extracts velocity at input pos and time
lDerivative = @(t,y,~)derivative(t,y,u_interpolant,v_interpolant, periodicBc, eps_domain);


%% LCS parameters
% Cauchy-Green strain rate tensor properties
cgEigenvalueFromMainGrid = true; % using aux grid may take too long 218 
cgAuxGridRelDelta = single(.01);
cgOdeSolverOptions = odeset('RelTol',1e-5,'initialStep',1e-6);
coupledIntegration = true;
method = 'finiteDifference'; % 'equationOfVariation'

Flow_Map = get_flowmap(lDerivative,domain,resolution,timespan,cgOdeSolverOptions);

[grad_Flow_Map11,grad_Flow_Map12,grad_Flow_Map21,grad_Flow_Map22] = get_flowmap_gradient(Flow_Map);



