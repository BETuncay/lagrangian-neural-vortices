
% boolean_mask_list: Return a binary mask of an list of elliptic lcs
%
% DESCRIPTION
% Given a closed line and its domain definition, calculate a logical matrix
% where the area encircled by the closed line is filled with true values.
%
% INPUT ARGUMENTS
% vhelp: vortex_helper object which contains the domain definiton and
% requirend conversion functions.
% lcs: (:, 2) double closed line
% show_plots: plot the end results and intermediate steps
%
% OUTPUT ARGUMENTS
% result: logical matrix with dimensions defined in vhelp
% true values represent points inside or on the boundary of the input lcs
function result = boolean_mask_list(vhelp, lcs_list, show_plots)
    
    res = [ceil(vhelp.res(1) / vhelp.sampling_rate), ceil(vhelp.res(2) / vhelp.sampling_rate)];
    result = false(res(1), res(2));
    for i = 1:numel(lcs_list)
       lcs = lcs_list{i};
       result = or(result, boolean_mask(vhelp, lcs, false));
    end
    if show_plots
        plot_mask(vhelp, result, 'result mask');
    end
end


% boolean_mask: Return a binary mask of an elliptic lcs
%
% DESCRIPTION
% Given a closed line and its domain definition, calculate a logical matrix
% where the area encircled by the closed line is filled with true values.
%
% INPUT ARGUMENTS
% vhelp: vortex_helper object which contains the domain definiton and
% requirend conversion functions.
% lcs: (:, 2) double closed line
% show_plots: plot the end results and intermediate steps
%
% OUTPUT ARGUMENTS
% result: logical matrix with dimensions defined in vhelp
% true values represent points inside or on the boundary of the input lcs
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
        x = randi([A(1), B(1)], npoints, 1);
        y = randi([A(2), B(2)], npoints, 1);
    
        [in, on] = inpolygon(x, y, line(:,1), line(:,2));
        in = find(in ~= on);
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

function [x, y] = bresenham(p1, p2)

%https://de.mathworks.com/matlabcentral/fileexchange/28190-bresenham-optimized-for-matlab
%Matlab optmized version of Bresenham line algorithm. No loops.
%Format:
%               [x y]=bham(x1,y1,x2,y2)
%
%Input:
%               (x1,y1): Start position
%               (x2,y2): End position
%
%Output:
%               x y: the line coordinates from (x1,y1) to (x2,y2)
%
%Usage example:
%               [x y]=bham(1,1, 10,-5);
%               plot(x,y,'or');


x1 = p1(1);
y1 = p1(2);
x2 = p2(1);
y2 = p2(2);

x1=round(x1); x2=round(x2);
y1=round(y1); y2=round(y2);
dx=abs(x2-x1);
dy=abs(y2-y1);
steep=abs(dy)>abs(dx);
if steep t=dx;dx=dy;dy=t; end

%The main algorithm goes here.
if dy==0 
    q=zeros(dx+1,1);
else
    q=[0;diff(mod([floor(dx/2):-dy:-dy*dx+floor(dx/2)]',dx))>=0];
end

%and ends here.

if steep
    if y1<=y2 y=[y1:y2]'; else y=[y1:-1:y2]'; end
    if x1<=x2 x=x1+cumsum(q);else x=x1-cumsum(q); end
else
    if x1<=x2 x=[x1:x2]'; else x=[x1:-1:x2]'; end
    if y1<=y2 y=y1+cumsum(q);else y=y1-cumsum(q); end
end
end