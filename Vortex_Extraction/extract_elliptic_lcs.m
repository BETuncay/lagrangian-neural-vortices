
% input flow field / path_num and config
% output lcs at certain points in time
function extract_elliptic_lcs(pathIn_num, config)


%% Input parameters
% start timer
timer_start = tic;

% construct vortex helper
pathIn_num_str = num2str(pathIn_num,'%04.f');
pathIn = ['E:\Fundue\', pathIn_num_str ,'.am'];
vhelp = vortex_helper(pathIn, config.sampling_rate);

% define domain
domain = [vhelp.domainMin(1),vhelp.domainMax(1);vhelp.domainMin(2),vhelp.domainMax(2)];
boundary_epsilon = 0.0002;
eps_domain = domain + [-boundary_epsilon, boundary_epsilon; -boundary_epsilon, boundary_epsilon];

% set main grid resolution
sampled_grid_resolution = get_sampled_resolution(vhelp, 1);
resolutionX = sampled_grid_resolution;
[resolutionY,~] = equal_resolution(domain,resolutionX);
resolution = [resolutionX,resolutionY];

%% Velocity field definition

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
cgEigenvalueFromMainGrid = true;
cgAuxGridRelDelta = single(.01);
cgOdeSolverOptions = odeset('RelTol',1e-5,'initialStep',1e-6);
coupledIntegration = true;
method = 'finiteDifference';

% Calculate elliptic LCS for each time specfied in initial_times
timespan = [config.extraction_times, config.extraction_times  + config.interval];
data_timespan = fix(transform_to_DataCoords(vhelp, timespan, 3));
for time = 1:numel(config.extraction_times)
    
%% Define Poincare Sections
% find pairs of points inside and outside of vortices --> vorticity maxima

% get velocity field at the start time
initial_flow = get_flow_vector_field2D(vhelp, data_timespan(time,1));

% calculate vorticity
vorticity = curl_2D(vhelp, initial_flow);

% extract vorticity maxima
threshold = 0.1;
neigbourhood_size = ceil(size(vorticity,1) / 10);
vorticity_maxima = local_maxima_bool_2(vhelp, vorticity, neigbourhood_size, threshold);

% point outside of a vortex --> point inside another vortex
% connect vorticity_maxima via minimum spanning tree
[~, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, vorticity_maxima);

% define poincare sections
poincareSection = struct('endPosition',{},'numPoints',{},'orbitMaxLength',{});
for i = 1:edgeSize
    start_point = edges(i, 1);
    end_point = edges(i, 2);
    poincareSection(i).endPosition = [pointsX(start_point),pointsY(start_point); pointsX(end_point),pointsY(end_point)];
end

% set amount of points to integrate on poincare section
[poincareSection.numPoints] = deal(config.poincareSection_numPoints);

% set max integration length before terminating
[poincareSection.orbitMaxLength] = deal(config.orbitMaxLength); % domain circumference = 4

%% Calculate Flow Map, Flow Map Gradient and Trajectory
% used as U-Net input and to calculate CG Tensor
trajectory_npoints = config.trajectory_numPoints;
Trajectory = get_trajectory(lDerivative,domain,resolution,timespan(time,:), trajectory_npoints, cgOdeSolverOptions);

Flow_Map = Trajectory(:,:,[1,2,trajectory_npoints*2-1,trajectory_npoints*2]); % first and last positions of the trajectory

[grad_Flow_Map11,grad_Flow_Map12,grad_Flow_Map21,grad_Flow_Map22] = get_flowmap_gradient(Flow_Map);

%% Cauchy-Green strain eigenvalues and eigenvectors
[cgEigenvector,cgEigenvalue] = eig_cgStrain(lDerivative,domain,resolution,timespan(time,:),'incompressible',incompressible,'eigenvalueFromMainGrid',cgEigenvalueFromMainGrid,'auxGridRelDelta',cgAuxGridRelDelta,'odeSolverOptions',cgOdeSolverOptions, 'coupledIntegration', coupledIntegration, 'method', method, 'Flow_Map', Flow_Map);
disp('Cauchy Green Strain Tensor calculated');

%% Elliptic LCSs

% define lambda parameter of eta fields, 
% too large/small lambda values --> complex values in the eta fields
% --> restricted lambda range such that no complex eta field values are possible
lambda_min = sqrt(max(cgEigenvalue(:,1)));
lambda_max = sqrt(min(cgEigenvalue(:,2)));
lambda = linspace(lambda_min, lambda_max, 4); % 
lambdaLineOdeSolverOptions = odeset('relTol',1e-5,'initialStep',1e-6);
forceEtaComplexNaN = true;
showPoincareGraph = false;

% calculate all losed lambda lines 
[closedLambdaLinePos,closedLambdaLineNeg] = poincare_closed_orbit_range(eps_domain,resolution,cgEigenvector,cgEigenvalue,lambda,poincareSection,'forceEtaComplexNaN',forceEtaComplexNaN,'odeSolverOptions',lambdaLineOdeSolverOptions, 'periodicBC', periodicBc, 'showPoincareGraph', showPoincareGraph);

% pick the most elliptic lambda line for each poincare section
ellipticLcs = [elliptic_lcs(closedLambdaLinePos, pointsX, pointsY, 'ellipse_area_linear_combination'), elliptic_lcs(closedLambdaLineNeg, pointsX, pointsY, 'ellipse_area_linear_combination')];

%stop timer
timer_end = toc(timer_start);
disp(timer_end);

% plot results
if config.plot_results  
    %plot_frle_psects(cgEigenvalue, ellipticLcs, closedLambdaLinePos, closedLambdaLineNeg, poincareSection, domain, resolution, timespan(time,:));
    ellipticColor = [0,.6,0];
    ellipticColor2 = [.6,0,0];

    [~, hAxes] = setup_figure(domain);
    title(hAxes,'elliptic LCSs')
    xlabel(hAxes,'x achse')
    ylabel(hAxes,'y achse')

    % Plot finite-time Lyapunov exponent
    cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
    ftle_ = ftle(cgEigenvalue2,diff(timespan));
    plot_ftle(hAxes,domain,resolution,ftle_);
    colormap(hAxes,gray)

    % Plot Poincare sections
    hPoincareSection = arrayfun(@(input)plot(hAxes,input.endPosition(:,1),input.endPosition(:,2)),poincareSection,'UniformOutput',false);
    hPoincareSection = [hPoincareSection{:}];
    set(hPoincareSection,'color',ellipticColor)
    set(hPoincareSection,'LineStyle','--')
    set(hPoincareSection,'marker','o')
    set(hPoincareSection,'MarkerFaceColor',ellipticColor)
    set(hPoincareSection,'MarkerEdgeColor','w')

    % Plot closed lambda lines
    hClosedLambdaLinePos = plot_closed_orbit(hAxes,closedLambdaLinePos);
    hClosedLambdaLineNeg = plot_closed_orbit(hAxes,closedLambdaLineNeg);
    hClosedLambdaLine = [hClosedLambdaLinePos,hClosedLambdaLineNeg];
    set(hClosedLambdaLine,'color',ellipticColor)

    % Plot elliptic LCSs
    hEllipticLcs = plot_elliptic_lcs(hAxes,ellipticLcs);
    set(hEllipticLcs,'color',ellipticColor2)
    set(hEllipticLcs,'linewidth',2)
    drawnow
end

% safe results into a mat file
if config.save_data
    current_path = pwd;
    if not(isfolder(config.save_folder_path))
        mkdir(config.save_folder_path)
    end
    cd(config.save_folder_path);
    
    flow_field_u = flow_field.u(data_timespan(time,1):1:data_timespan(time,1) + 5, :, :);
    flow_field_v = flow_field.v(data_timespan(time,1):1:data_timespan(time,1) + 5, :, :);
    lcs_mask = boolean_mask_list(vhelp, ellipticLcs, config.plot_results);
    
    time_str = num2str(config.extraction_times(time));
    interval_str = num2str(config.interval);
    file_name = char(['lcs_', pathIn_num_str, '_', time_str, '_', interval_str, '.mat']);
        
    save(file_name,'flow_field_u', 'flow_field_v', 'lcs_mask', 'vorticity_maxima', 'ellipticLcs', 'closedLambdaLinePos', 'closedLambdaLineNeg', 'cgEigenvector', 'cgEigenvalue', 'Flow_Map', 'Trajectory', 'poincareSection', 'grad_Flow_Map11', 'grad_Flow_Map12', 'grad_Flow_Map21', 'grad_Flow_Map22', 'timer_end', 'pathIn_num_str', 'time_str', 'interval_str'); 
    
    cd(current_path);
    drawnow

end
end
end