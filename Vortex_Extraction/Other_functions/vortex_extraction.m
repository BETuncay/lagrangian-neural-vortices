% this was the first file created in this thesis
% it is included for historical reasons :)
%% Cleanup
clear;
close all;
clc;

%% Parameters
%pathIn = 'D:\_Tools_Data\Matlab_Data\Vortices Dataset\1000.am';
pathIn = 'G:\Fundue\3000.am'; % von 0000 bis 7995, jedes 5te element
sampling_rate = 1;
sampling_rate_time = 1;

%% Code
vhelp = vortex_helper(pathIn, sampling_rate);
grid = get_mesh_grid2D(vhelp);
treshold = 0.8;

flow_field = get_flow_vector_field2D(vhelp, 400);

vorticity = curl_2D(vhelp, flow_field);
abs_vorticity = abs(vorticity); 

vorticity_maxima = local_maxima_bool_2(vhelp, vorticity, 101, treshold);
Tvortex = vorticity_treshold(vhelp, vorticity, treshold);

draw_vorticity2D(vhelp, abs_vorticity);

hold on
[pointsY, pointsX] = find(vorticity_maxima);
pointsX = transform_to_DomainCoords(vhelp, pointsX, 1);
pointsY = transform_to_DomainCoords(vhelp, pointsY, 2);
plot(pointsX, pointsY, 'r*', 'color', 'r');


draw_vorticity2D(vhelp, Tvortex);
%colormap gray

%% Math functions

function local_max = deprecated_local_maxima_bool(vorticity)
   
    shape = size(vorticity);
    local_max = logical(zeros(shape));
    for i = 2:(shape(1) - 1)
        for j = 2: (shape(2) - 1)
           neighbours = [vorticity(i+1, j+1), vorticity(i+1, j), vorticity(i+1, j-1),
                          vorticity(i, j+1), vorticity(i, j), vorticity(i, j-1),
                          vorticity(i-1, j+1), vorticity(i-1, j), vorticity(i-1, j-1)];
           if vorticity(i, j) >= neighbours
               local_max(i,j) = true;
           elseif vorticity(i, j) <= neighbours
               local_max(i,j) = true;
           end
        end
    end
    
    for i = 1:(shape(1) - 1):shape(1)
        for j = 1:(shape(2) - 1):shape(2)
            down = mod(i-2,shape(1))+1;
            left = mod(j-2,shape(2))+1;
            up = mod(i,shape(1))+1;
            right = mod(j,shape(2))+1;
           neighbours = [vorticity(up, right), vorticity(up, j), vorticity(up, left),
                          vorticity(i, right), vorticity(i, j), vorticity(i, left),
                          vorticity(down, right), vorticity(down, j), vorticity(down, left)];
           if vorticity(i, j) >= neighbours
               local_max(i,j) = true;
           elseif vorticity(i, j) <= neighbours
               local_max(i,j) = true;
           end
        end
    end
    
end

function filter = deprecated_filter_points(vorticity, bool_vort)

    changed = true;
    dims = size(vorticity);
    abs_vorticity = abs(vorticity);
    while changed
        changed = false;
        [row,col] = find(bool_vort);
        for i = 1:length(row)
            down = mod(row(i)-2,dims(1))+1;
            left = mod(col(i)-2,dims(2))+1;
            up = mod(row(i),dims(1))+1;
            right = mod(col(i),dims(2))+1;
            neighbours = abs_vorticity(up, left) + abs_vorticity(up, col(i)) + abs_vorticity(up, right) + abs_vorticity(row(i), left) + abs_vorticity(row(i), right)+ abs_vorticity(down, left) + abs_vorticity(down, col(i)) + abs_vorticity(down, right);
            if 8 * abs_vorticity(row(i), col(i)) < neighbours
                bool_vort(row(i), col(i)) = false;
                changed = true;
            end
        end
    end
    filter = bool_vort;
end

function J = jacobian2D(u, v)
    
    J = [partial_diff(u, 1), partial_diff(u, 2)
         partial_diff(v, 1), partial_diff(v, 2)];
end

function interp = nearest_neighbour(flow_field, xi, yi, step, sampling_rate)

    u = flow_field.u;
    v = flow_field.v;
    size = length(u); % size of u and v (u, v are n x n)

    x = transform_to_DataCoords(xi, sampling_rate);
    x1 = fix(x);
    x2 = x1 + 1;
    x_frac = x - x1;
    y = transform_to_DataCoords(yi, sampling_rate);
    y1 = fix(y);
    y2 = y1 + 1;
    y_frac = y - y1;
    
    
    if x1 == 0 % right edge of domain
        x1 = size;
        x2 = 1;
    elseif x1 == size
        x2 = 1;
    end
    
    if y1 == 0 % right edge of domain
        y1 = size;
        y2 = 1;
    elseif y1 == size
        y2 = 1;
    end
    
    if x_frac <= 0.5
        x = x1;
    else
        x = x2;
    end
    
    if y_frac <= 0.5
        y = y1;
    else
        y = y2;
    end
    

    interp = [u(y, x), v(y, x)];
end

function interp = no_interpolate(flow_field, xi, yi, step, sampling_rate)
    
    u = flow_field.u;
    v = flow_field.v;
    size = length(u); % size of u and v (u, v are n x n)

    x = fix(transform_to_DataCoords(xi, sampling_rate));
    y = fix(transform_to_DataCoords(yi, sampling_rate));

    
    if x == 0 % right edge of domain
        x = size;
    elseif x == size
        x = 1;
    end
    
    if y == 0 % right edge of domain
        y = size;
    elseif y == size
        y = 1;
    end
   
    interp = [u(y, x), v(y, x)];
end

function interp = bilinear_interpolate(flow_field, xi, yi, step, sampling_rate)

    u = flow_field.u;
    v = flow_field.v;
    size = length(u); % size of u and v (u, v are n x n)
    
    x = transform_to_DataCoords(xi, sampling_rate);
    x1 = fix(x);
    x2 = x1 + 1;
    y = transform_to_DataCoords(yi, sampling_rate);
    y1 = fix(transform_to_DataCoords(yi, sampling_rate));
    y2 = y1 + 1;
    
    if x1 == 0 % right edge of domain
        x1 = size;
        x2 = 1;
    elseif x1 == size
        x2 = 1;
        x = x - size;
    end
    
    if y1 == 0 % right edge of domain
        y1 = size;
        y2 = 1;
    elseif y1 == size
        y2 = 1;
        y = y - size;
    end
    
    A = [1, x1, y1, x1 * y1;
         1, x1, y2, x1 * y2;
         1, x2, y1, x2 * y1;
         1, x2, y2, x2 * y2];
     
    corners_u = [u(y1, x1); u(y2, x1); u(y1, x2); u(y2, x2)];
    corners_v = [v(y1, x1); v(y2, x1); v(y1, x2); v(y2, x2)];
    weights_u = linsolve(A, corners_u);
    weights_v = linsolve(A, corners_v);
    
    interp_u = dot([1; x; y; x * y], weights_u);
    interp_v = dot([1; x; y; x * y], weights_v);
    
    interp = [interp_u, interp_v];
    
end

function y = eulerMethod(u, initialPos, step, endValue, sampling_rate)
    % Euler's Method
    % Initial conditions and setup
    h = step;  % step size
    t = 0:step:endValue;  % the range of t
    y = zeros(size(t));  % allocate the result y
    y(1) = initialPos;  % the initial y value
    n = numel(y);  % the number of y values
    % The loop to solve the DE
    for i=1:n-1
        
        yi = transform_to_DataCoords(y(i), sampling_rate);
        yi_int = fix(yi);
        yi_fract = yi - yi_int;
        
        u_interp = u(yi_int) * (1 - yi_fract);
        if yi_int == ceil(512/sampling_rate) % right edge of domain
            u_interp = u_interp + u(1) * yi_fract; 
        else
            u_interp = u_interp + u(yi_int + 1) * yi_fract;
        end
        
        y(i+1) = y(i) + h * u_interp;
    end
end

function [x, y] = eulerMethod2D(flow_field, initialPos, step, endValue, interp_function, sampling_rate)
    % Euler's Method
    % Initial conditions and setup
    h = step;  % step size
    %t = initialValue:h:endValue;
    t = 0:h:endValue; 
    x = zeros(size(t));
    y = zeros(size(t));  
    x(1) = initialPos(1);
    y(1) = initialPos(2);  
    n = numel(y);  
    % The loop to solve the DE
    for i=1:n-1
        
        diff = interp_function(flow_field, x(i), y(i), step, sampling_rate);
        x(i+1) = x(i) + h * diff(1);
        y(i+1) = y(i) + h * diff(2);
    end
end

function [x, y] = runge_kutta2D(flow_field, initialPos, step, endValue, interp_function, sampling_rate)
    % runge_kutta 4 order
    % Initial conditions and setup
    h = step;  % step size
    %t = initialValue:h:endValue;
    t = 0:h:endValue; 
    x = zeros(size(t));
    y = zeros(size(t));  
    x(1) = initialPos(1);
    y(1) = initialPos(2);  
    n = numel(y);  
    % The loop to solve the DE
    for i=1:n-1
        
        X = x(i);
        Y = y(i);
        k1 = interp_function(flow_field, X, Y, step, sampling_rate);
        
        X = x(i) + 0.5 * h * k1(1);
        Y = y(i) + 0.5 * h * k1(2);
        k2 = interp_function(flow_field, X, Y, step, sampling_rate);
        
        X = x(i) + 0.5 * h * k2(1);
        Y = y(i) + 0.5 * h * k2(2);
        k3 = interp_function(flow_field, X, Y, step, sampling_rate);
        
        X = x(i) + h * k3(1);
        Y = y(i) + h * k3(2);
        k4 = interp_function(flow_field, X, Y, step, sampling_rate);
        
        x(i+1) = x(i) + 0.1667 * h * (k1(1) + 2 * k2(1) + 2 * k3(1) + k4(1));
        y(i+1) = y(i) + 0.1667 * h * (k1(2) + 2 * k2(2) + 2 * k3(2) + k4(2));
    end
end

% calculate streamline with seed \in [0, 1] for distance steps
function [pointsX, pointsY] = calculate_streamline(flow_field, seed, step, distance, sampling_rate)

    % set starting value
    distance = fix((distance - 2) / step) + 1; % normalize step, step count same but step = 1
    distance = distance + 2 - 1;
    
    pointsX = zeros(1, distance); % array of points x-cord
    pointsY = zeros(1, distance); % array of points y-cord
    pointsX(1)= seed(1);          % set seed x-cord
    pointsY(1) = seed(2);         % set seed y-cord
    pos = [seed(1), seed(2)];     % current point used to calculate next points
    dataSize = ceil(512/sampling_rate);
    
    u = flow_field.u;
    v = flow_field.v;
    
    % get the points of a streamline
    
    for tau = 2:1:distance
        % convert pos into dataSpace
        x = transform_to_DataCoords(pos(1), sampling_rate);
        y = transform_to_DataCoords(pos(2), sampling_rate);
        
        
        % linear linterpolate the value from the 4 surounding grid points
        xfrac = mod(x, 1);
        yfrac = mod(y, 1);
        x = fix(x);
        y = fix(y);
        xn = mod(x, dataSize) + 1;
        yn = mod(y, dataSize) + 1;
        
        % fix for weird behaviour, where fix(1) = 0 or something else
        if x == 0
            x = dataSize;
        end
        if y == 0
            y = dataSize;
        end
        
        u_values = u(y, x) * (1 - xfrac) + u(y, xn) * xfrac;
        v_values = v(y, x) * (1 - yfrac) + v(yn, x) * yfrac;
        
        % get velocity at x, y from vector field and add to pos
        diff = [u_values, v_values];
        pos = pos + diff * step;
        
        % append new point
        pointsX(tau) = pos(1);
        pointsY(tau) = pos(2);
    end
end
%% Drawing functions


function draw = draw_streamline(pointsX, pointsY)

    length = size(pointsX);
    length = length(2);
    plot3(pointsX, pointsY, zeros(1, length), 'o-','MarkerSize', 3, 'MarkerFaceColor', 'red','MarkerIndices', 1); 
end

% draw the 2D unsteady vector field going through time
function draw_vector_field_overtime = draw_vector_field_overtime(data, sampling_rate, pause_time)
    
    grid = get_mesh_grid2D(sampling_rate);
    for t = 1:1:1001
        flow_field = get_flow_vector_field2D(data, sampling_rate, t);
        draw_vector_field2D(flow_field, grid)
        pause(pause_time);
    end
end

%% Testing functions

function test = draw_streamline_test(data, sampling_rate)
    
    time = 500;
    seed_distance = 0.2; % seed_distance \in (0, 1)
    step = 0.1;
    distance = 2;
    pause_time = 0.1;
    
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);

    figure
    draw_vector_field2D(flow_field, grid);
    hold on
    for i = 0:seed_distance:1
        for j = 0:seed_distance:1
            seed = [j, i];
            [pointsX, pointsY] = calculate_streamline(flow_field, seed, step, distance, sampling_rate);
            draw_streamline(pointsX, pointsY);
            pause(0.1);
        end
    end
    hold off
end

function test = draw_eulerMethod2D(data, sampling_rate)
    
    time = 500;
    seed_distance = 0.2; % seed_distance \in (0, 1)
    step = 0.1;
    distance = 3;
    pause_time = 0.1;
    interp = @bilinear_interpolate;
    
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);

    figure
    draw_vector_field2D(flow_field, grid);
    hold on
    title('Euler Method Test')
    for i = 0:seed_distance:1
        for j = 0:seed_distance:1
            initialPos = [j, i];
            [pointsX, pointsY] = eulerMethod2D(flow_field, initialPos, step, distance, interp, sampling_rate);
            draw_streamline(pointsX, pointsY);
            pause(pause_time);
        end
    end
    hold off
end

function test = draw_compareEulerInterpolation(data, sampling_rate)
    
    time = 500;
    step = 0.01;
    initialPos = [0.5 , 0.5];
    distance = 10;
    
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);

    [x, y] = eulerMethod2D(flow_field, initialPos, step, distance, @no_interpolate, sampling_rate);
    figure
    hold on
    title('No Interpolation')
    draw_vector_field2D(flow_field, grid);
    draw_streamline(x, y);
    
    [x, y] = eulerMethod2D(flow_field, initialPos, step, distance, @nearest_neighbour, sampling_rate);
    figure
    hold on
    title('Nearest Neighbour')
    draw_vector_field2D(flow_field, grid);
    draw_streamline(x, y);
    
    [x, y] = eulerMethod2D(flow_field, initialPos, step, distance, @bilinear_interpolate, sampling_rate);
    figure
    hold on
    title('Bilinear Interpolation ')
    draw_vector_field2D(flow_field, grid);
    draw_streamline(x, y);

end

function test = draw_runge_kutta2D(data, sampling_rate)
    
    time = 500;
    seed_distance = 0.2; % seed_distance \in (0, 1)
    step = 0.1;
    distance = 3;
    pause_time = 0.1;
    interp = @bilinear_interpolate;
    
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);

    figure
    draw_vector_field2D(flow_field, grid);
    hold on
    title('Runge Kutta Test')
    for i = 0:seed_distance:1
        for j = 0:seed_distance:1
            initialPos = [j, i];
            [pointsX, pointsY] = runge_kutta2D(flow_field, initialPos, step, distance, interp, sampling_rate);
            draw_streamline(pointsX, pointsY);
            pause(pause_time);
        end
    end
    hold off
end

function test = ndgradient_test(data, sampling_rate)
   
    time = 1000;
    u = squeeze(data(1, 1:sampling_rate:512, 1:sampling_rate:512, time));
    [x, y] = meshgrid(0:sampling_rate/511:1, 0:sampling_rate/511:1);
    
    Du = ndgradient(u);
    
    figure('Name','Function');
    mesh(x, y, u);
    
    figure('Name','Derivative');
   
    hold on
    contour(x, y, u)
    quiver(x, y, Du{1}, Du{2})
    hold off
    
end

function test = curl2D_test(data, sampling_rate)

    time = 10;
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);
    
    vorticity = curl_2D(flow_field.u, flow_field.v);
    %vorticity = abs(vorticity);
    
    figure
    draw_vector_field2D(flow_field, grid);
    
   
    figure
    imagesc([0, 1], [0, 1], vorticity);
    set(gca,'YDir','normal') 
    colorbar;
end

function test = flow_field3D_test(data, sampling_rate)
    

    test_time = randi(ceil(1001 / sampling_rate), 1);

    flow_field3D = get_flow_vector_field3D(data, sampling_rate); 
    grid3D = get_mesh_grid3D(sampling_rate);
    
    flow_field2D = get_flow_vector_field2D(data, sampling_rate, test_time); 
    grid2D = get_mesh_grid2D(sampling_rate);
    
    figure
    axis equal
    quiver3(grid3D{1}, grid3D{2}, grid3D{3}, flow_field3D{1},flow_field3D{2},flow_field3D{3});
    
    figure
    axis equal
    u = flow_field3D{1}; 
    v = flow_field3D{2};
    u = u(:, :, test_time);
    v = v(:, :, test_time);
      
    quiver(grid2D{1}, grid2D{2}, u, v, 1);
    
    figure
    axis equal
    quiver(grid2D{1}, grid2D{2}, flow_field2D{1}, flow_field2D{2}, 1);
end

function test = adaptToLCSTest()

    load('ocean_geostrophic_velocity.mat');

    length = size(lat);
    Lon = transpose(repmat(lon, [1, length(1)]));
    Lat = repmat(lat, [1, length(1)]);

    VLon = squeeze(vLon(1,:,:));
    VLat = squeeze(vLat(1,:,:));

    figure
    quiver(Lon, Lat, VLon, VLat, 1);


    [X, Y] = meshgrid(0:sampling_rate/511:1, 0:sampling_rate/511:1);
    u = squeeze(data(2, 1:sampling_rate:512, 1:sampling_rate:512, 1:sampling_rate:1001));
    v = squeeze(data(1, 1:sampling_rate:512, 1:sampling_rate:512, 1:sampling_rate:1001));

    u = squeeze(u(:,:,1));
    v = squeeze(v(:,:,1));

    figure
    quiver(X, Y, u, v, 1);

    % Lon = X
    % Lat = Y

    xy = transpose(0:sampling_rate/511:1);
end

function test = optimizationFkts(data, sampling_rate)
    
    time = 10;
    % get flow field and grid
    flow_field = get_flow_vector_field2D(data, sampling_rate, time);
    grid = get_mesh_grid2D(sampling_rate);

    % get vorticity and its deriviative
    vorticity = curl_2D(flow_field.u, flow_field.v);

    % threshold variant
    treshold = 0.1;
    Tvorts = vorticity_treshold(vorticity, treshold);

    lmax_conv = local_maxima_conv(vorticity);
    lmax_bool = local_maxima_bool(vorticity);

    lmax_filter = filter_semi_definite(lmax_bool, vorticity);

    figure
    quiver(grid.x,grid.y, flow_field.u, flow_field.v);

    figure
    title('Vorticity Threshold')
    surf(grid.x, grid.y, vorticity);
    hold on
    plot3(grid.x(Tvorts),grid.y(Tvorts),vorticity(Tvorts),'r*', 'color', 'r');
    
    figure
    title('Vorticity Maxima Convolution')
    surf(grid.x, grid.y, vorticity);
    hold on
    plot3(grid.x(lmax_conv),grid.y(lmax_conv),vorticity(lmax_conv),'r*', 'color', 'r');

    figure
    title('Vorticity Maxima Bool')
    surf(grid.x, grid.y, vorticity);
    hold on
    plot3(grid.x(lmax_bool),grid.y(lmax_bool),vorticity(lmax_bool),'r*', 'color', 'r');

    figure
    title('Vorticity Maxima Filtered Bool');
    surf(grid.x, grid.y, vorticity);
    hold on
    plot3(grid.x(lmax_filter),grid.y(lmax_filter),vorticity(lmax_filter),'r*', 'color', 'r');
    
    figure
    imagesc([0, 1], [0, 1], vorticity);
    set(gca,'YDir','normal') 
    colorbar;
end


function test = noise_reduction_visualization2D()

fig = figure;
amplitude = 0.1;
for step = 0.001:0.001:1%linspace(0.001, 2, 500)
x = -2:step:2;
y = x .^2;
y_noise = y + amplitude * rand(1, length(y));
plot(x, y_noise);
axis([-1,1,0,1]);
axis square;
s = sprintf('step size=%0.001f',step);
ht = text(0,0.75,s,'Color', 'red', 'FontSize', 15);
drawnow;
pause(0.1);
end


end


function test = noise_reduction_visualization3D()
pathIn = 'G:\Fundue\1000.am'; % von 0000 bis 7995, jedes 5te element
sampling_rate = 1;
vhelp = vortex_helper(pathIn, sampling_rate);
time = transform_to_DataCoords(vhelp, 5, 3);
fig = figure;
for i = 1:50
    vhelp.sampling_rate = i;
    initial_flow = get_flow_vector_field2D(vhelp, time);
    vorticity = curl_2D(vhelp, initial_flow);
    vorticity = imgaussfilt(vorticity, 2, 'Padding', 'circular');
    local_max_points = local_maxima_bool_2(vhelp, vorticity, 5);
    %draw_vorticity3D(vhelp, vorticity, fig);
    draw_vorticity_maxima(vhelp, vorticity, local_max_points, fig); 
    axis([0,1,0.25,1,-7,7]);
    axis square;
    s = sprintf('sampling rate=%0.1f',i);
    ht = text(0.25,0,-6,s,'Color', 'red', 'FontSize', 15);
    pause(2)
end
end
