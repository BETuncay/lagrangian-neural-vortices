% collection of many different plots used in the thesis
clc
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set';
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);
domain = [vhelp.domainMin(1),vhelp.domainMax(1);vhelp.domainMin(2),vhelp.domainMax(2)];

sampled_grid_resolution = get_sampled_resolution(vhelp, 1);
resolutionX = sampled_grid_resolution;
[resolutionY,~] = equal_resolution(domain,resolutionX);
resolution = [resolutionX,resolutionY];

ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];
%% SET PLOT NUMBER 
image_nr = 11;
save_image = true;
%%
for i = 4:size(listing, 1)
    result = listing(i).name;
    load(result);
    close all
    time = str2double(time_str);
    interval = str2double(interval_str);
    timespan = [time, time+interval];
    % plot vorticity with the poincare sections overlayed
    flow_field.u = squeeze(flow_field_u(1,:,:));
    flow_field.v = squeeze(flow_field_v(1,:,:));
    vorticity = curl_2D(vhelp, flow_field);
    disp(result)
    
    
    try 
    [span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, vorticity_maxima);
    catch ME
        fprintf('%s\n', ME.message);
    end
    % define poincare sections
    poincareSection = struct('endPosition',{},'numPoints',{},'orbitMaxLength',{});
    for j = 1:edgeSize
        start_point = edges(j, 1);
        end_point = edges(j, 2);
        poincareSection(j).endPosition = [pointsX(start_point),pointsY(start_point); pointsX(end_point),pointsY(end_point)];
    end

    fig = figure;
    fig.Position = [0 150 1000 800];
    tiledlayout(1,1)
    %sgtitle(result,'Interpreter','none');

    
  
    if isempty(ellipticLcs)
        continue
    end
    if image_nr == 1
        ax1 = nexttile;
        % Plot finite-time Lyapunov exponent
        cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
        ftle_ = ftle(cgEigenvalue2,diff(timespan));
        plot_ftle(ax1,domain,resolution,ftle_);
        colormap(ax1,gray)
        %title(ax1,'elliptic LCSs')
        set(gca,'YDir','normal');

        hold on
        % Plot Poincare sections
        %for psec = 1:size(poincareSection,2)
        %    pos = poincareSection(psec).endPosition;
        %    plot(ax1,pos(:,1),pos(:,2), 'Color', ellipticColor, 'LineStyle','--','marker','o','MarkerFaceColor',ellipticColor,'MarkerEdgeColor','w');
        %end

        % Plot elliptic LCSs
        hEllipticLcs = plot_elliptic_lcs(ax1,ellipticLcs);
        set(hEllipticLcs,'color',ellipticColor)
        set(hEllipticLcs,'linewidth',3)
        %%%
    end
    
    if image_nr == 2
    
        ax2 = nexttile;
        lcs = ellipticLcs{1};
        res = [ceil(vhelp.res(1) / vhelp.sampling_rate), ceil(vhelp.res(2) / vhelp.sampling_rate)];
        lcs = ceil(transform_to_DataCoords(vhelp, lcs, 1));
        lcs = [lcs; lcs(1,:)];
        lcs_extended = lcs + [res(1), res(2)];
        result_extended = false(res(1) * 3, res(2) * 3);
        result_extended = draw_closed_line(result_extended, lcs_extended);
        [result_extended, x, y, in] = fill_closed_lines(result_extended, lcs_extended, 100);

        imagesc([1,1536], [1, 1536], result_extended);
        set(gca,'YDir','normal')
        colormap(ax2,gray)
        hold on
        linewidth = 3;
        color = [1,1,1];
        
        plot(ax2,[512,512],[1,1536], 'Color', color, 'LineStyle','-','LineWidth',linewidth);
        plot(ax2,[1024,1024],[1,1536], 'Color', color, 'LineStyle','-','LineWidth',linewidth);
        plot(ax2,[1,1536], [512,512], 'Color', color, 'LineStyle','-','LineWidth',linewidth);
        plot(ax2,[1,1536], [1024,1024], 'Color', color, 'LineStyle','-','LineWidth',linewidth);

    end
    
    if image_nr == 3
    
        ax3 = nexttile;
        lcs = ellipticLcs{1};
        res = [ceil(vhelp.res(1) / vhelp.sampling_rate), ceil(vhelp.res(2) / vhelp.sampling_rate)];
        lcs = ceil(transform_to_DataCoords(vhelp, lcs, 1));
        lcs = [lcs; lcs(1,:)];
        lcs_extended = lcs + [res(1), res(2)];
        result_extended = false(res(1) * 3, res(2) * 3);
        result_extended = draw_closed_line(result_extended, lcs_extended);
        [result_extended, x, y, in] = fill_closed_lines(result_extended, lcs_extended, 100);
        result = collapse_matrix(res, result_extended);
        
        imagesc([1,512], [1, 512], result);
        set(gca,'YDir','normal')
        colormap(ax3,gray)   
    end
    
    if image_nr == 4
        
        vorticity = repmat(vorticity,3);
        vorticity_height = zeros(size(ellipticLcs{1}, 1));
        vorticity_height = vorticity_height + 0.1;

        [x, y] = meshgrid(linspace(-1,2,1536), linspace(-1,2,1536));
        h = surf(x, y, vorticity);
        set(h,'LineStyle','none');
        title('Vorticity');
        hold on
        plot3(ellipticLcs{1}(:,1),ellipticLcs{1}(:,2),vorticity_height(:,1))
        %draw_vorticity2D(vhelp, vorticity, fig, plot_title)
    end
    
    if image_nr == 5
        
        vorticity = repmat(vorticity,3);

        imagesc([-1, 2], [-1, 2], vorticity);
        set(gca,'YDir','normal') 
        title('Vorticity');
        hold on
        plot(ellipticLcs{1}(:,1),ellipticLcs{1}(:,2))
        
        linewidth = 3;
        color = [0,0,0];

        plot([0,0],[-1,2], 'Color', color,'LineWidth',linewidth);
        plot([1,1],[-1,2], 'Color', color,'LineWidth',linewidth);
        plot([-1,2], [0,0], 'Color', color,'LineWidth',linewidth);
        plot([-1,2], [1,1], 'Color', color,'LineWidth',linewidth);
    end
    
    if image_nr == 6
        threshold = 0.1;
        neigbourhood_size = ceil(size(vorticity,1) / 10);
        vorticity_maxima = local_maxima_bool_2(vhelp, vorticity, neigbourhood_size, threshold);
        
        imagesc([0, 1], [0, 1], abs(vorticity));
        set(gca,'YDir','normal')
        
        hold on
        [pointsY, pointsX] = find(vorticity_maxima);
        pointsX = transform_to_DomainCoords(vhelp, pointsX, 1);
        pointsY = transform_to_DomainCoords(vhelp, pointsY, 2);
        %plot(pointsX, pointsY, 'LineStyle','none','EdgeColor', [1,0,0], 'NodeColor', [1,0,0]);
        sz = 200;
        scatter(pointsX,pointsY,sz,'MarkerEdgeColor',[.3 .2 .2],...
              'MarkerFaceColor',[1 0 0],...
              'LineWidth',1)
    end
    
    if image_nr == 7
        pathIn = 'E:\Fundue\0550.am';
        vhelp = vortex_helper(pathIn, 30);
        time_rate = 100;
        flow_field.u = squeeze(vhelp.data(2, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:time_rate:vhelp.res(3)));
        flow_field.v = squeeze(vhelp.data(1, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:time_rate:vhelp.res(3)));
              
        lengthx = ceil(vhelp.res(1) / vhelp.sampling_rate);
        lengthy = ceil(vhelp.res(2) / vhelp.sampling_rate);
        lengthz = ceil(vhelp.res(3) / time_rate);
        flow_field.t = single(zeros(lengthx, lengthy, lengthz));
        
        [x, y, z] = meshgrid(0:(vhelp.sampling_rate * (1 - 0)) / (vhelp.res(1) - 1):1, ...
        0:(vhelp.sampling_rate * (1 - 0)) / (vhelp.res(2) - 1):1, ...
        0:(time_rate * (10 - 0)) / (vhelp.res(3) - 1):10);
    
        step = 5;
        q = quiver3(x(:,:,1:step:end), y(:,:,1:step:end), z(:,:,1:step:end), ...
        flow_field.u(:,:,1:step:end), flow_field.v(:,:,1:step:end), flow_field.t(:,:,1:step:end));
        
        set(gca,'DataAspectRatio',[1 1 6])
        q.ShowArrowHead = 'off';
        q.Marker = '.';
        q.MarkerFaceColor = [0.8,0.8,0.8];
        
        xlabel('X-Axis')
        ylabel('Y-Axis')
        zlabel('Time')
    end
    
    if image_nr == 8
        pathIn = 'E:\Fundue\0550.am';
        vhelp = vortex_helper(pathIn, 30);
        time_rate = 100;
        flow_field.u = squeeze(vhelp.data(2, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:time_rate:vhelp.res(3)));
        flow_field.v = squeeze(vhelp.data(1, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:time_rate:vhelp.res(3)));
              
        lengthx = ceil(vhelp.res(1) / vhelp.sampling_rate);
        lengthy = ceil(vhelp.res(2) / vhelp.sampling_rate);
        lengthz = ceil(vhelp.res(3) / time_rate);
        flow_field.t = single(zeros(lengthx, lengthy, lengthz));
        
        [x, y, z] = meshgrid(0:(vhelp.sampling_rate * (1 - 0)) / (vhelp.res(1) - 1):1, ...
        0:(vhelp.sampling_rate * (1 - 0)) / (vhelp.res(2) - 1):1, ...
        0:(time_rate * (10 - 0)) / (vhelp.res(3) - 1):10);
    
        step = 5;
        q = quiver3(x(:,:,1:step:end), y(:,:,1:step:end), z(:,:,1:step:end), ...
                flow_field.u(:,:,1:step:end), flow_field.v(:,:,1:step:end), flow_field.t(:,:,1:step:end));
        
        set(gca,'DataAspectRatio',[1 1 6])
        %q.ShowArrowHead = 'off';
        %q.Marker = '.';
        q.Color = [0.55,0.55,0.55];
        q.MarkerFaceColor = [0.8,0.8,0.8];
        
        %xlabel('X-Axis')
        %ylabel('Y-Axis')
        %zlabel('Time')
        
        % trajectory
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

        timespan = [0,10];  
        trajectory_npoints = 100;
        times = linspace(0,10,trajectory_npoints);
        trajectory = get_trajectory(lDerivative,domain,resolution,timespan,trajectory_npoints,SolverOptions);
        epsdomain = domain;% + [-0.1, 0.1; -0.1, 0.1];
        
        rng('default')
        hold on
        cmap = colormap;
        particle_step = 1;
        for y = 1:particle_step:resolution(1)
            for x = 1:particle_step:resolution(2)
                xPoints = transpose(squeeze(trajectory(y,x,1:2:end)));
                yPoints = transpose(squeeze(trajectory(y,x,2:2:end)));
                minX = find(or(xPoints < epsdomain(1,1), xPoints > epsdomain(1,2)));
                minY = find(or(yPoints < epsdomain(2,1), yPoints > epsdomain(2,2)));
                maxHeight = min([minX, minY]) * 2;
                if isempty(maxHeight)
                    if rand(1,1) > 0.55
                        p = plot3(transpose(squeeze(trajectory(y,x,1:2:end))), transpose(squeeze(trajectory(y,x,2:2:end))), times);
                        p.LineWidth = 1.2;
                        p.Color = cmap(mod(x*5 + y*5,255)+1,:);
                    end
                else
                    if maxHeight >= 500
                        p =plot3(transpose(squeeze(trajectory(y,x,1:2:maxHeight))), transpose(squeeze(trajectory(y,x,2:2:maxHeight))), times(1:(maxHeight/2)));
                        p.LineWidth = 1.2;
                        p.Color = cmap(mod(x*5 + y*5,255)+1,:);
                    end
                end
            end
        end
    end
    

    if image_nr == 10
        ax1 = nexttile;
        draw_vorticity2D(vhelp, lcs_mask, fig, 'Binary Mask');
        colormap(ax1,gray)
    end
    
    
    if image_nr == 11
        ax3 = nexttile;
        cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
        ftle_ = ftle(cgEigenvalue2,diff(timespan));
        plot_ftle(ax3,domain,resolution,ftle_);
        colormap(ax3,gray)
        title(ax3,'elliptic LCSs')
        set(gca,'YDir','normal') 

        hold on

        % Plot closed lambda lines
        hClosedLambdaLinePos = plot_closed_orbit(ax3,closedLambdaLinePos);
        hClosedLambdaLineNeg = plot_closed_orbit(ax3,closedLambdaLineNeg);
        set(hClosedLambdaLinePos, 'color',ellipticColor)
        set(hClosedLambdaLineNeg, 'color',ellipticColor)

        % Plot elliptic LCSs
        hEllipticLcs = plot_elliptic_lcs(ax3,ellipticLcs);
        set(hEllipticLcs,'color',ellipticColor)
        set(hEllipticLcs,'linewidth',1)
        
    end
    
    if save_image
        iptsetpref('ImshowBorder','tight');
        set(gca,'visible','off')
        saveas(gcf,'filename.png')
    end
    drawnow
end




function result = boolean_mask(vhelp, lcs, show_plots)
    
    res = [ceil(vhelp.res(1) / vhelp.sampling_rate), ceil(vhelp.res(2) / vhelp.sampling_rate)];

    % transform lcs_line into data coordinates [0,1]^2 -> [1, 512]^2
    lcs = ceil(transform_to_DataCoords(vhelp, lcs, 1));
    
    % append first point to lcs to close possible gaps
    lcs = [lcs; lcs(1,:)];

    % extend lcs_line to handle lines outside of the domain
    lcs_extended = lcs + [res(1), res(2)];

    % allocate extended result matrix ( extended for cases outside of the domain
    result_extended = false(res(1) * 3, res(2) * 3);

    % draw the elliptic lcs boundary lines into the result matrix
    result_extended = draw_closed_line(result_extended, lcs_extended);

    % fill the inside of the elliptic lcs
    [result_extended, x, y, in] = fill_closed_lines(result_extended, lcs_extended, 100);

    % condense the mask back into its original size
    result = collapse_matrix(res, result_extended);

    %plot results
    if show_plots
        plot_point_cloud(lcs_extended, x, y, in);
        plot_mask(vhelp, result_extended, 'extended filled lcs');
        plot_mask(vhelp, result, 'filled lcs');
    end
end


function matrix = draw_closed_line(matrix, line)
    
    matrix_size = size(matrix);
    for i = 1:length(line) - 1
        [x, y] = bresenham(line(i,:), line(i+1,:));       
        ind = sub2ind(matrix_size, y, x);
        matrix(ind) = true;

    end
end

function [matrix, x, y, in] = fill_closed_lines(matrix, line, npoints)
    
    % get bounding box
    A = min(line); 
    B = max(line); 

    idx = [];
    while isempty(idx)
        x = randi([A(1), B(1)],npoints, 1);
        y = randi([A(2), B(2)],npoints, 1);
    
        [in, ~] = inpolygon(x, y, line(:,1), line(:,2));
        idx = find(in, 1);
        matrix = imfill(matrix, [y(idx), x(idx)], 4);
    end
end

function [matrix, x, y, in] = fill_closed_lines_old(matrix, line, npoints)
    
    % get bounding box
    A = min(line); 
    B = max(line); 

    x = randi([A(1), B(1)],npoints, 1);
    y = randi([A(2), B(2)],npoints, 1);
    
    [in, ~] = inpolygon(x, y, line(:,1), line(:,2));
    idx = find(in, 1);
    matrix = imfill(matrix, [y(idx), x(idx)], 4);
end

function matrix = collapse_matrix(res, matrix)

    main_x = res(1)+1:res(1)*2;
    main_y = res(2)+1:res(2)*2;
    for i = 0:2
        t1 = (1 + i * res(2)):((i + 1) * res(2));
        for j = 0:2
            t2 = (1 + j * res(1)):((j + 1) * res(1));
            matrix(main_y, main_x) =  or (matrix(main_y, main_x), matrix(t1, t2));
        end
    end

    matrix = matrix(main_y, main_x);
    
%unrolled loop version: 
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(1:512, 1:512));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(1:512, 513:512*2));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(1:512, 512*2+1:512*3));

%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(513:512*2, 1:512));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(513:512*2, 513:512*2));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(513:512*2, 512*2+1:512*3));

%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(512*2+1:512*3, 1:512));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(512*2+1:512*3, 513:512*2));
%ground_truth_filled(main, main) =  or (ground_truth_filled(main, main), ground_truth_filled(512*2+1:512*3, 512*2+1:512*3));
    
end

function plot_mask(vhelp, matrix, name)
    figure
    imagesc([vhelp.domainMin(1), vhelp.domainMax(1)], [vhelp.domainMin(2), vhelp.domainMax(2)], matrix);
    title(name);
    set(gca,'YDir','normal');
    colorbar;
end

function plot_point_cloud(line, x, y, in)

    figure
    plot(line(:,1), line(:,2), x(in), y(in),'.r', x(~in),y(~in),'.b');
end


function [T, Y] = get_tracetory(derivative,domain,resolution,timesteps,odeSolverOptions)

    initialPosition = initialize_ic_grid(resolution,domain);
    [T,Y] = ode45_vector(@(t,y)derivative(t,y,false),timesteps,initialPosition,false,odeSolverOptions);
    

end

function [tf, yf] = ode45_vector(odefun,tspan,y0,useEoV,options)

% Reshape m-by-2 array to column array
y0 = transpose(y0);
y0 = y0(:);

if useEoV
    coupledSize = 6;
else
    coupledSize = 2;
end

% Specify three timesteps for ode45's tspan. This has been reported to
% reduce memory usage.

[tf,yf] = ode45(odefun,tspan,y0,options);
yf = yf(:);
yf = transpose(reshape(yf,size(tspan)*2,size(yf,2)/coupledSize));
end