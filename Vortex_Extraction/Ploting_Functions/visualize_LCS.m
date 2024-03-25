% function used to plot elliptic and hyperbolic LCS

clc
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean';
cd(result_path);
listing = dir(result_path);

%% PICK WHICH HYPERBOLIC LCS TO PLOT
plot_repelling_LCS = true;
plot_attracting_LCS = false;

for i = 103:size(listing, 1)
    result = listing(i).name;
    load(result);
    close all
    time = str2double(time_str);
    interval = str2double(interval_str);
    timespan = [time, time+interval];
    % plot vorticity with the poincare sections overlayed

res = result(5:8);
pathIn = ['E:\Fundue\', res ,'.am'];
sampling_rate = 10;
vhelp = vortex_helper(pathIn, sampling_rate);

domain = [vhelp.domainMin(1),vhelp.domainMax(1);vhelp.domainMin(2),vhelp.domainMax(2)];
boundary_epsilon = 0.0002;
eps_domain = domain + [-boundary_epsilon, boundary_epsilon; -boundary_epsilon, boundary_epsilon];

sampled_grid_resolution = get_sampled_resolution(vhelp, 1);
resolutionX = sampled_grid_resolution;
[resolutionY,deltaX] = equal_resolution(domain,resolutionX);
resolution = [resolutionX,resolutionY];

ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];


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

cgEigenvalueFromMainGrid = true; % using aux grid may take too long
cgAuxGridRelDelta = single(.01);
cgOdeSolverOptions = odeset('RelTol',1e-5,'initialStep',1e-6);
coupledIntegration = true;
method = 'finiteDifference';

% define function lDerivative: extracts velocity at input pos and time
lDerivative = @(t,y,~)derivative(t,y,u_interpolant,v_interpolant, periodicBc, eps_domain);
    %% Calculate Flow Map and its derivative
Flow_Map = get_flowmap(lDerivative,domain,resolution,timespan,cgOdeSolverOptions);

%% Cauchy-Green strain eigenvalues and eigenvectors
[cgEigenvector,cgEigenvalue] = eig_cgStrain(lDerivative,domain,resolution,timespan,'incompressible',incompressible,'eigenvalueFromMainGrid',cgEigenvalueFromMainGrid,'auxGridRelDelta',cgAuxGridRelDelta,'odeSolverOptions',cgOdeSolverOptions, 'coupledIntegration', coupledIntegration, 'method', method, 'Flow_Map', Flow_Map);
disp('Cauchy Green Strain Tensor calculated');
    
% Shrink lines
shrinkLineMaxLength = 50;
shrinkLineLocalMaxDistance = 2*deltaX;
shrinkLineOdeSolverOptions = odeset('relTol',1e-6);

% Stretch lines
stretchLineMaxLength = 50;
stretchLineLocalMaxDistance = 4*deltaX;
stretchLineOdeSolverOptions = odeset('relTol',1e-6);

% Graphic properties
repellingColor = [0.78, 0, 0];
attractingColor = [0.08, 0.6, 0.9];
ellipticColor = [0,.6,0];
initialPositionMarkerSize = 2;

[hFigure, hAxes] = setup_figure(domain);
%title(hAxes,'Repelling and elliptic LCSs')
%xlabel(hAxes,'Longitude (\circ)')
%ylabel(hAxes,'Latitude (\circ)')


% Plot finite-time Lyapunov exponent
cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
ftle_ = ftle(cgEigenvalue2,diff(timespan));
plot_ftle(hAxes,domain,resolution,ftle_);
colormap(hAxes,gray)
set(gca,'YDir','normal') 
drawnow

%% Elliptic LCSs

% Plot elliptic LCSs
hEllipticLcs = plot_elliptic_lcs(hAxes,ellipticLcs);
set(hEllipticLcs,'color',ellipticColor);
set(hEllipticLcs,'linewidth',3);
drawnow

nPoincareSection = size(ellipticLcs, 2);
%% Hyperbolic repelling LCSs
if plot_repelling_LCS
[shrinkLine,shrinkLineInitialPosition] = seed_curves_from_lambda_max(shrinkLineLocalMaxDistance,shrinkLineMaxLength,cgEigenvalue(:,2),cgEigenvector(:,1:2),domain,resolution,'odeSolverOptions',shrinkLineOdeSolverOptions);

% Remove shrink lines inside elliptic LCSs
for j = 1:nPoincareSection
    shrinkLine = remove_strain_in_elliptic(shrinkLine,ellipticLcs{j});
    idx = inpolygon(shrinkLineInitialPosition(1,:),shrinkLineInitialPosition(2,:),ellipticLcs{j}(:,1),ellipticLcs{j}(:,2));
    shrinkLineInitialPosition = shrinkLineInitialPosition(:,~idx);
end

% Plot hyperbolic repelling LCSs
hRepellingLcs = cellfun(@(position)plot(hAxes,position(:,1),position(:,2)),shrinkLine,'UniformOutput',false);
hRepellingLcs = [hRepellingLcs{:}];
set(hRepellingLcs,'color',repellingColor)
set(hRepellingLcs,'linewidth',1);
hShrinkLineInitialPosition = arrayfun(@(idx)plot(hAxes,shrinkLineInitialPosition(1,idx),shrinkLineInitialPosition(2,idx)),1:size(shrinkLineInitialPosition,2),'UniformOutput',false);
hShrinkLineInitialPosition = [hShrinkLineInitialPosition{:}];

uistack(hEllipticLcs,'top')
end
%% Hyperbolic attracting LCSs
if plot_attracting_LCS

[stretchLine,stretchLineInitialPosition] = seed_curves_from_lambda_max(stretchLineLocalMaxDistance,stretchLineMaxLength,-cgEigenvalue(:,1),cgEigenvector(:,3:4),domain,resolution,'odeSolverOptions',stretchLineOdeSolverOptions);

% Remove stretch lines inside elliptic LCSs
for j = 1:nPoincareSection
    stretchLine = remove_strain_in_elliptic(stretchLine,ellipticLcs{j});
    idx = inpolygon(stretchLineInitialPosition(1,:),stretchLineInitialPosition(2,:),ellipticLcs{j}(:,1),ellipticLcs{j}(:,2));
    stretchLineInitialPosition = stretchLineInitialPosition(:,~idx);
end

% Plot hyperbolic attracting LCSs
hAttractingLcs = cellfun(@(position)plot(hAxes,position(:,1),position(:,2)),stretchLine,'UniformOutput',false);
hAttractingLcs = [hAttractingLcs{:}];
set(hAttractingLcs,'color',attractingColor);
set(hAttractingLcs,'linewidth',1);
hStretchLineInitialPosition = arrayfun(@(idx)plot(hAxes,stretchLineInitialPosition(1,idx),stretchLineInitialPosition(2,idx)),1:size(stretchLineInitialPosition,2),'UniformOutput',false);
hStretchLineInitialPosition = [hStretchLineInitialPosition{:}];

uistack(hEllipticLcs,'top')
end
end