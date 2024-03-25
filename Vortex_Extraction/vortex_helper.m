classdef vortex_helper
    properties
        % default values based on the flow fields defined in: https://cgl.ethz.ch/publications/papers/paperJak20a.php
        pathIn,
        data,
        sampling_rate (1,1) {mustBePositive, mustBeInteger, mustBeNonzero} = 1,
        domainMin (:, 1) {mustBeNumeric} = [0.9766e-3; 0.9766e-3; 0],
        domainMax (:, 1) {mustBeNumeric} = [0.9990230; 0.9990230; 10],
        res (:,1) {mustBePositive, mustBeInteger, mustBeNonzero} = [512; 512; 1001],
        numComponents (1,1) {mustBePositive, mustBeInteger, mustBeNonzero} = 2,
        gradient_step = 0.1
    end
    methods
        %% Constructor
        function obj = vortex_helper(pathIn, sampling_rate, read_amira)
            arguments
                pathIn {mustBeText} = 'G:\Fundue\1000.am'
                sampling_rate {mustBePositive, mustBeInteger, mustBeNonzero} = 1
                read_amira (1,1) logical = true
            end
            if nargin < 2
                disp('The following 2 inputs are required: datapath and sampling_rate')
            end
                obj.pathIn = pathIn;
                obj.sampling_rate = sampling_rate;
                
            if read_amira
                [data, domainMin, domainMax, res, numComponents] = readAmira(pathIn);
                obj.data = data;
                obj.domainMin = double(domainMin);
                obj.domainMax = double(domainMax);
                obj.res = double(res); % anpassen mit sampling rate
                obj.numComponents = numComponents;        
                obj.gradient_step = 0.1;
            end
        end
     
        %% Data Getters
        % Get flow field at a set point in time.
        function flow_field = get_flow_vector_field2D(obj, time, sampling_rate)
        arguments
            obj vortex_helper
            time (1,1) {mustBePositive, mustBeInteger, mustBeNonzero}
            sampling_rate (1,1) {mustBePositive, mustBeInteger, mustBeNonzero} = obj.sampling_rate
        end
            flow_field.u = squeeze(obj.data(2, 1:sampling_rate:obj.res(1), 1:sampling_rate:obj.res(2), ceil(time / sampling_rate)));
            flow_field.v = squeeze(obj.data(1, 1:sampling_rate:obj.res(1), 1:sampling_rate:obj.res(2), ceil(time / sampling_rate)));
        end
     
        % Get flow field
        function flow_field = get_flow_vector_field3D(obj, include_time)
            arguments
                obj vortex_helper
                include_time = false
            end
         
            flow_field.u = squeeze(obj.data(2, 1:obj.sampling_rate:obj.res(1), 1:obj.sampling_rate:obj.res(2), 1:obj.sampling_rate:obj.res(3)));
            flow_field.v = squeeze(obj.data(1, 1:obj.sampling_rate:obj.res(1), 1:obj.sampling_rate:obj.res(2), 1:obj.sampling_rate:obj.res(3)));
            if include_time
                lengthx = ceil(obj.res(1) / obj.sampling_rate);
                lengthy = ceil(obj.res(2) / obj.sampling_rate);
                lengthz = ceil(obj.res(3) / obj.sampling_rate);
                flow_field.t = single(ones(lengthx, lengthy, lengthz));
            end
        end
        
        % Get 1D grid fitting to the flow field at dimension dim
        function grid = get_mesh_grid1D(obj, dim, epsilon)
            arguments
                obj vortex_helper
                dim
                epsilon = 0
            end
            
            domainMin = obj.domainMin - [epsilon; epsilon; 0];
            domainMax = obj.domainMax + [epsilon; epsilon; 0];
            % if we assume obj.domainMin(1) = 0 and obj.domainMax(1) = 1
            %grid = 0:obj.sampling_rate/(obj.res(1)-1):1;
            grid = domainMin(dim):(obj.sampling_rate * (domainMax(dim) - domainMin(dim))) / (obj.res(dim) - 1):domainMax(dim);
            grid = single(grid);
        end
    
        % Get 2D grid fitting to the flow field (x,y - Dimension)
        function grid = get_mesh_grid2D(obj)
            arguments
                obj vortex_helper
            end 
            [x, y] = meshgrid(obj.domainMin(1):(obj.sampling_rate * (obj.domainMax(1) - obj.domainMin(1))) / (obj.res(1) - 1):obj.domainMax(1), ...
            obj.domainMin(2):(obj.sampling_rate * (obj.domainMax(2) - obj.domainMin(2))) / (obj.res(2) - 1):obj.domainMax(2));
            grid.x = x;
            grid.y = y;
        end
     
        % Get 3D grid fitting to the flow field
        function grid = get_mesh_grid3D(obj)
            arguments
                obj vortex_helper
            end          
            [x, y, z] = meshgrid(obj.domainMin(1):(obj.sampling_rate * (obj.domainMax(1) - obj.domainMin(1))) / (obj.res(1) - 1):obj.domainMax(1), ...
            obj.domainMin(2):(obj.sampling_rate * (obj.domainMax(2) - obj.domainMin(2))) / (obj.res(2) - 1):obj.domainMax(2), ...
            obj.domainMin(3):(obj.sampling_rate * (obj.domainMax(3) - obj.domainMin(3))) / (obj.res(3) - 1):obj.domainMax(3));
            grid.x = x;
            grid.y = y;
            grid.z = z;
        end
         
        % transforms [domMin, domMax] --> {1, ... ,N}
        % !!! fractions are keept in the result e.g. interpolation
        function dataX = transform_to_DataCoords(obj, x, dim)
            arguments
                obj vortex_helper
                x {mustBeNumeric} % value to interpolate
                dim {mustBeNumeric} % dimension of x
            end 
            N = ceil(obj.res(dim) / obj.sampling_rate);
            startPoint = [obj.domainMin(dim), 1];
            endPoint = [obj.domainMax(dim), N];
            dataX = interpolate_line(obj, x, startPoint, endPoint);
        end

        % transforms [1, ... ,N] --> [0, 1]
        function dataX = transform_to_DomainCoords(obj, x, dim)
            arguments
                obj vortex_helper
                x {mustBeNumeric} % value to interpolate
                dim {mustBeNumeric} % dimension of x
            end 
        N = ceil(obj.res(dim) / obj.sampling_rate);
        startPoint = [1, obj.domainMin(dim)];
        endPoint = [N, obj.domainMax(dim)];
        dataX = interpolate_line(obj, x, startPoint, endPoint);
        end
      
        % get resolution along input dimension
        function sampled_res = get_sampled_resolution(obj, dim)
            arguments
                obj vortex_helper
                dim {mustBeNumeric} % dimension
            end 
            sampled_res = ceil(obj.res(dim) / obj.sampling_rate);
        end
        
        % interpolate value of a point on the line defined by the start and end point
        function interp_value = interpolate_line(obj, query_point, start_point, end_point)
            arguments
                obj vortex_helper
                query_point {mustBeNumeric}
                start_point (2,1) {mustBeNumeric} % point = [x_value, y_value]
                end_point (2,1) {mustBeNumeric}
            end
        
            samples.x = [start_point(1), end_point(1)];
            samples.y = [start_point(2), end_point(2)];

            interp_value = interp1(samples.x, samples.y, query_point, 'linear', 'extrap');
        end
       
        %% Linear Operators
        
        % Calculates the numerical gradient of n-dim input value with
        % standard spacing
        function grad = ndgradient(obj, function_values)
            arguments
                obj vortex_helper
                function_values {mustBeNumeric}
            end
            
            grad = cell(ndims(function_values), 1);
            [grad{:}] = gradient(function_values, obj.gradient_step);
        end
        
        % Calculates the numerical gradient of 2-dim input value with
        % given grid spacing
        function [gradX, gradY] = flow_gradient(obj, function_values, gridX, gridY)
            arguments
                obj vortex_helper
                function_values (:,:) {mustBeNumeric}
                gridX {mustBeNumeric} = get_mesh_grid1D(obj, 1)
                gridY {mustBeNumeric} = get_mesh_grid1D(obj, 2)
            end
         
            [gradX, gradY] = gradient(function_values, gridX, gridY);
        end
     
        % Calculates the curl of input flow field struct
        function curl_2D = curl_2D(obj, flow_field)
            arguments
                obj vortex_helper
                flow_field
            end
            [~, gradUy] = flow_gradient(obj, flow_field.u);
            [gradVx, ~] = flow_gradient(obj, flow_field.v);
            curl_2D = gradVx - gradUy;
        end
        
        % Calculates the divergence of input flow field struct
        function divergence = divergence_2D(obj, flow_field)
            arguments
                obj vortex_helper
                flow_field
            end
            [gradUx, ~] = flow_gradient(obj, flow_field.u);
            [~, gradVy] = flow_gradient(obj, flow_field.v);
            divergence = gradUx + gradVy;

        end

        %% Optimization Functions / Vortex Detection
        
        % Calculate the vorticity threshold and return a boolean matrix
        % threshold is a value between 0 and 1
        function Tvortex = vorticity_treshold(obj, vorticity, threshold)
            arguments
                obj vortex_helper
                vorticity (:,:) {mustBeNumeric}
                threshold (1,1) {mustBeNumeric}
                
            end
            max_vort = max(abs(vorticity), [], 'all', 'linear');
            min_vort = min(abs(vorticity), [], 'all', 'linear');
            threshold = (max_vort - min_vort) * threshold;
            Tvortex = abs(vorticity) > threshold;
        end
     
        function Zeros = naive_zeros(obj, vorticity, epsilon)
            arguments
                obj vortex_helper
                vorticity (:,:) {mustBeNumeric}
                epsilon (1,1) {mustBeNumeric}
                
            end
            [vortDx, vortDy] = flow_gradient(obj, vorticity);
            % naive alternative with extremas
            ZerosX = (- epsilon < vortDx) & (vortDx < epsilon);
            ZerosY = (- epsilon < vortDy) & (vortDy < epsilon);
            Zeros = ZerosX & ZerosY;
        end
     
        function local_max = local_maxima_conv(obj, vorticity, kernel_size)
            arguments
                obj vortex_helper
                vorticity (:,:) {mustBeNumeric}
                kernel_size (1,1) {mustBeNumeric}  
            end
            
            % kernel size must be odd to have a middle point
            if not(mod(kernel_size, 2))
                kernel_size = kernel_size + 1;
            end
            
            % pad input matrix with a = padding_layers such that dim (n+a)x(m+a) 
            padding_layers = floor(kernel_size * 0.5);
            padded_vorticity = padarray(vorticity,[padding_layers padding_layers],'circular');
            padded_vorticity = abs(padded_vorticity);
            
            kernel = ones(kernel_size, kernel_size);
            neighbours = conv2(padded_vorticity, kernel, 'valid');
            num_elem = kernel_size * kernel_size;
            local_max = (abs(vorticity) * num_elem > neighbours);
        end
     
        function local_max = local_maxima_bool_2(obj, vorticity, kernel_size, threshold)
            arguments
                obj vortex_helper
                vorticity (:,:) {mustBeNumeric}
                kernel_size (1,1) {mustBeNumeric}
                threshold (1,1) {mustBeNumeric}
            end
            
            % kernel size must be odd to have a middle point
            if not(mod(kernel_size, 2))
                kernel_size = kernel_size + 1;
            end
            
            % get maximum vorticity value
            max_vort = max(abs(vorticity), [], 'all', 'linear');
            min_vort = min(abs(vorticity), [], 'all', 'linear');
            
            threshold = (max_vort - min_vort) * threshold;
            %disp(threshold);
            
            % pad input matrix with a padding_layer such that dim (n+a)x(m+a)
            % inorder to compare neighbours in the period domain
            padding_layers = floor(kernel_size * 0.5);
            padded_vorticity = padarray(vorticity,[padding_layers padding_layers],'circular');
            
            % compute absolute value of input matrix
            padded_vorticity = abs(padded_vorticity);
            mid_point = padding_layers + 1; % index of kernel middle point

            shape = size(vorticity);
            local_max = false(shape); % logical result matrix
            
            % keep track of regions where a maximum has already been found
            % only one maximum per region possible -> skip search
            found_region = false(size(padded_vorticity));
            
            for i = mid_point:shape(1) + padding_layers
                for j = mid_point:shape(2) + padding_layers
                    
                    % point cant be a maximum
                    if found_region(i,j)
                        continue
                    end
                    
                    a = padded_vorticity(i, j);
                    % skip if value is below threshold
                    if a < threshold
                        continue
                    end
                    
                    neighbours = padded_vorticity(i-padding_layers:i+padding_layers, j-padding_layers:j+padding_layers);
                    neighbours(mid_point,mid_point) = 0;
                    B = a >= neighbours;
                    
                    if all(B)
                        %disp(a);
                        local_max(i - padding_layers,j - padding_layers) = true;
                        found_region(i-padding_layers:i+padding_layers, j-padding_layers:j+padding_layers) = 1;
                    end
                end
            end
        end
        
        function local_max = local_maxima_bool(obj, vorticity)
         
            shape = size(vorticity);
            local_max = false(shape);
            
            % create matrix with (n+2)x(m+2) where the input matrix is inserted
            abs_vorticity = padarray(vorticity,[1 1],'circular');
            abs_vorticity = abs(abs_vorticity);
         
            for i = 2:shape(1)+1
                for j = 2:shape(2)+1
                    neighbours = [abs_vorticity(i + 1, j + 1), abs_vorticity(i + 1, j), abs_vorticity(i + 1, j - 1), ...
                    abs_vorticity(i, j + 1),                     abs_vorticity(i, j - 1), ...
                    abs_vorticity(i - 1, j + 1), abs_vorticity(i - 1, j), abs_vorticity(i - 1, j - 1)];
                    if all(abs_vorticity(i, j) > neighbours)
                        local_max(i - 1, j - 1) = true;
                    end               
                end
            end
        end
        
        function local_max = local_maxima_region(obj, vorticity, region_count)
            
            shape = size(vorticity);
            local_max = false(shape);
            
            region_size = 2 ^ region_count;
            
            stepsX = floor(obj.res(1) / region_size);
            stepsY = floor(obj.res(2) / region_size);
            
            %abs_vorticity = repmat(abs(vorticity), 2);
            abs_vorticity = abs(vorticity);

            
            for i = 0:stepsY - 1
                si = 1 + i * region_size;
                ti = (i + 1) * region_size;
                t1 = si:ti;
                
                for j = 0:stepsX - 1
                    sj = 1 + j * region_size;
                    tj = (j + 1) * region_size;
                    t2 = sj:tj;

                    [~, index] = min(abs_vorticity(t1,t2), [], 'all', 'linear');
                    [row,col] = ind2sub([region_size, region_size], index);
                    local_max(row + si - 1, col + sj - 1) = 1;
                end
            end
        end
        
        function filtered_bool = filter_semi_definite(obj, vorticity, bool_vort)

        filtered_bool = bool_vort;
        [vortDx, vortDy] = flow_gradient(obj, vorticity);
        [vortDxx, vortDxy] = flow_gradient(obj, vortDx);
        [vortDyx, vortDyy] = flow_gradient(obj, vortDy);
        %Dvorticity = mygradient(obj, vorticity);
        %A = mygradient(obj, Dvorticity{1});
    	%B = mygradient(obj, Dvorticity{2});
    
        [row,col] = find(bool_vort);
        for i = 1:length(row)
            Hessian = [vortDxx(row(i),col(i)), vortDxy(row(i),col(i));
                       vortDyx(row(i),col(i)), vortDyy(row(i),col(i))];
            eigenvalues = sign(eig(Hessian));
    
            if eigenvalues(1) ~= eigenvalues(2)
               filtered_bool(row(i),col(i)) = false;
            end
        end
        end
        
        %% Graph Operations
        % calculate the minimum spanning tree of a fully connected metric
        % graph defined by the input points
        function minspantreee = min_spantree(obj, pointsX, pointsY)
    
        npoints = length(pointsX);
        idxs = nchoosek(1:npoints,2);
        dist = hypot(pointsY(idxs(:,1)) - pointsY(idxs(:,2)), pointsX(idxs(:,1)) - pointsX(idxs(:,2)));
        G = graph(idxs(:,1),idxs(:,2), dist);
     
        [minspantreee,pred] = minspantree(G);
        end

        % returns the boundary point that is closest to the inputpoints
        function [boundary_point, closest_point_idx] = get_closest_point_to_boundary(obj, pointsX, pointsY)
        
        % check point count    
        if numel(pointsX) == 0
            boundary_point =  [0,0];
            closest_point_idx = 0;
            return
        end
           
        if numel(pointsX) == 1
            boundary_point =  [obj.domainMax(1), obj.domainMax(2) * 0.5];
            closest_point_idx = 1;
            return
        end
        
        % convex hull needs minimum 3 points --> add one point
        if numel(pointsX) == 2
            midPointX = (obj.domainMax(1) - obj.domainMin(1)) / 2 + obj.domainMin(1);
            midPointY = (obj.domainMax(2) - obj.domainMin(2)) / 2 + obj.domainMin(2);
            distances = hypot(pointsX - midPointX, pointsY - midPointY);
            [~,idx] = max(distances);
            
            closest_point_idx = idx;
            
            cross_points = cell(1, 4);
            cross_points{1} = [obj.domainMax(1), pointsY(idx)];
            cross_points{2} = [obj.domainMin(1), pointsY(idx)];
            cross_points{3} = [pointsX(idx), obj.domainMax(2)];
            cross_points{4} = [pointsX(idx), obj.domainMin(2)];
           
            minDistance = inf;
            for i = 1:4
                distance = hypot(pointsX(idx) - cross_points{i}(1), pointsY(idx) - cross_points{i}(2));
                if distance < minDistance
                    minDistance = distance;
                    boundary_point = cross_points{i};
                end      
            end
            return
        end
        
        % lastly add point at closest boundary
        hull = convhull(pointsX,pointsY);
        closest.value = 9999;

        for i = 1:length(hull)
            distances = [pointsX(hull(i)) - obj.domainMin(1), obj.domainMax(1) - pointsX(hull(i)), ...
                pointsY(hull(i)) - obj.domainMin(2), obj.domainMax(2) - pointsY(hull(i))];
            [value, idx] = min(distances);
            if value < closest.value
            closest.point = hull(i);
            closest.value = value;
            closest.dir = idx;
            end
        end

        boundary_point = [0,0];
        switch closest.dir
            case 1
            boundary_point =  [obj.domainMin(1), pointsY(closest.point)];
            case 2
            boundary_point =  [obj.domainMax(1), pointsY(closest.point)];
            case 3
            boundary_point =  [pointsX(closest.point), obj.domainMin(2)];
            case 4
            boundary_point =  [pointsX(closest.point), obj.domainMax(2)];
        end
        
        closest_point_idx = closest.point;
        
        end
        
        
        % returns the boundary point that is furthest away to the input
        function boundary_point = get_furtherest_point_to_boundary(obj, pointsX, pointsY)
            cross_points = cell(1, 4);
            cross_points{1} = [obj.domainMax(1), obj.domainMax(2) * 0.5];
            cross_points{2} = [obj.domainMin(1), obj.domainMax(2) * 0.5];
            cross_points{3} = [obj.domainMax(1) * 0.5, obj.domainMax(2)];
            cross_points{4} = [obj.domainMax(1) * 0.5, obj.domainMin(2)];
           
            maxDistance = -inf;
            for i = 1:4
                distance = hypot(pointsX - cross_points{i}(1), pointsY - cross_points{i}(2));
                if distance > maxDistance
                    maxDistance = distance;
                    boundary_point = cross_points{i};
                end      
            end
        
        end
        
        
        % turn undirected connected graph into directed connected graph
        % describing the poincare sections
        % https://en.wikipedia.org/wiki/Algebraic_connectivity
        function poincare_edges = build_poincare_sections(obj, edges, method)
            
            switch(method)
                % reflect all edges and add them to the set of edges
                % remove all edges with non-unique starting vertex
                % --> this approach may lead to a nonconnected graph 
                case('reflect_add_remove')
                    poincare_edges = [edges; [edges(:,2), edges(:,1)]];
                    [~,ia] = unique(poincare_edges(:,1));
                    poincare_edges = poincare_edges(ia, :);
                    
                % if the starting vertex of an edges is not unique 
                % -> flip all edges where the starting_vertex is not unique
                % if all starting vertices unique -> add missing vertex
                case('mirror_loop')
                    
                    max_vertex = max(edges, [], 'all', 'linear');
                    a = find(edges(:,2) == max_vertex);
                    if not(isempty(a))
                        edges(a,:) = [edges(a,2), edges(a,1)];  
                    end
                    while length(edges(:,1)) ~= length(unique(edges(:,1)))
                        [~, w] = unique(edges(:,1), 'stable'); 
                        duplicate_indices = setdiff( 1:numel(edges(:,1)), w );
                        edges(duplicate_indices,:) = [edges(duplicate_indices,2), edges(duplicate_indices,1)];                       
                    end
                    disp(edges);
                    edges = sort(edges);
                    disp(edges);
                    % add missing point
                    b = 1:max_vertex;
                    disp(edges);
                    
                    
                    % reflect all edges and add them to the set of edges
                    % -> every vertex has minimum one edge starting on it
                    % will take longer to calculate, maybe more robust
                    case('reflect_add')
                        poincare_edges = [edges; [edges(:,2), edges(:,1)]];
            end
            

        end
         
        function [span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(obj, local_max_points) 
        arguments
            obj vortex_helper
            local_max_points (:,:)
        end
                
        [pointsY, pointsX] = find(local_max_points);
        pointsX = transform_to_DomainCoords(obj, pointsX, 1);
        pointsY = transform_to_DomainCoords(obj, pointsY, 2);
        
        if numel(pointsX) == 0
            span_tree = graph; 
            edges = 0;
            edgeSize = 0;
            return   
        end
        
        if numel(pointsX) == 1
            [boundary_point, ~] = get_closest_point_to_boundary(obj, pointsX, pointsY);
            pointsX = [pointsX; boundary_point(1)];
            pointsY = [pointsY; boundary_point(2)];
        end
        
        span_tree = min_spantree(obj, pointsX, pointsY);

        edges = span_tree.Edges.EndNodes;
        edges = build_poincare_sections(obj, edges, 'reflect_add');
        
        % add one edge to closest boundary point
        [boundary_point, closest_point_idx] = get_closest_point_to_boundary(obj, pointsX, pointsY);
        pointsX = [pointsX; boundary_point(1)];
        pointsY = [pointsY; boundary_point(2)];
        edges = [edges; closest_point_idx , size(pointsX, 1)];
        span_tree = addedge(span_tree, closest_point_idx, size(pointsX, 1), 1);
        edgeSize = size(edges, 1);
        end
        
        %% Drawing Functions
        
        function draw_flow_field2D(obj, flow_field, fig)
        arguments
            obj vortex_helper
            flow_field (:,:)
            fig matlab.ui.Figure = figure
        end
            grid = get_mesh_grid2D(obj);
            quiver(grid.x, grid.y, flow_field.u, flow_field.v);
            title('Flow Field 2D')
        end
        
        function draw_flow_field3D(obj, flow_field, fig)
        arguments
            obj vortex_helper
            flow_field (:,:,:)
            fig matlab.ui.Figure = figure
        end
            grid = get_mesh_grid3D(obj);
            quiver3(grid.x, grid.y, grid.z, flow_field.u, flow_field.v, flow_field.t, 2);
            title('Flow Field 3D')
        end
        
        function draw_vorticity3D(obj, vorticity, fig)
        arguments
            obj vortex_helper
            vorticity (:,:)
            fig matlab.ui.Figure = figure
        end
        figure(fig);
        
        grid = get_mesh_grid2D(obj);
        h = surf(grid.x, grid.y, vorticity);
        set(h,'LineStyle','none');
        title('Vorticity');
        end
        
        function draw_vorticity2D(obj, vorticity, fig, plot_title)
        arguments
            obj vortex_helper
            vorticity (:,:)
            fig matlab.ui.Figure = figure
            plot_title = 'Vorticity'
        end
        figure(fig);

        imagesc([obj.domainMin(1), obj.domainMax(1)], [obj.domainMin(2), obj.domainMax(2)], vorticity);
        set(gca,'YDir','normal') 
        %colorbar;
        title(plot_title);
        end
        
        function draw_vorticity_maxima3D(obj, vorticity, lmax, fig, plot_title)
        arguments
            obj vortex_helper
            vorticity (:,:)
            lmax (:,:)
            fig matlab.ui.Figure = figure
            plot_title = 'Vorticity Maxima'
        end
        figure(fig);
        
        grid = get_mesh_grid2D(obj);
        h = surf(grid.x, grid.y, vorticity);
        set(h,'LineStyle','none')
        hold on
        plot3(grid.x(lmax),grid.y(lmax),vorticity(lmax),'r*', 'color', 'r');
        title(plot_title);
        hold off
        end
        
        function draw_vorticity_maxima2D(obj, vorticity, lmax, fig, plot_title)
        arguments
            obj vortex_helper
            vorticity (:,:)
            lmax (:,:)
            fig matlab.ui.Figure = figure
            plot_title = 'Vorticity Maxima'
        end
        figure(fig);
        draw_vorticity2D(obj, vorticity, fig);

        hold on
        [pointsY, pointsX] = find(lmax);
        pointsX = transform_to_DomainCoords(obj, pointsX, 1);
        pointsY = transform_to_DomainCoords(obj, pointsY, 2);
        plot(pointsX, pointsY, 'r*', 'color', 'r');
        title(plot_title);
        hold off
        end
        
        function draw_poincare_sections(obj, vorticity, span_tree, pointsX, pointsY, fig, plot_title)
        arguments
            obj vortex_helper
            vorticity (:,:)
            span_tree
            pointsX
            pointsY
            fig matlab.ui.Figure = figure
            plot_title = 'Poincare Sections'
        end
        figure(fig);
        draw_vorticity2D(obj, vorticity, fig)
        hold on;
        plot(span_tree, 'XData', pointsX, 'YData', pointsY, 'NodeLabel',{}, 'EdgeColor', 'r', 'NodeColor', 'r', 'LineWidth', 1.5, 'MarkerSize', 5);
        title(plot_title);
        hold off;
        end
        
    end
end