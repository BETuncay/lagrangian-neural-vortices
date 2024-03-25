%% Cleanup
clear;
close all;
clc;

%% Parameters

pathIn = 'E:\Fundue\0110.am';
sampling_rate = 1;
sampling_rate_time = 1;
treshold = 0.1;

%% Code
vhelp = vortex_helper(pathIn, sampling_rate);
grid = get_mesh_grid2D(vhelp);

time = 7;
time = fix(transform_to_DataCoords(vhelp, time, 3));
flow_field = get_flow_vector_field2D(vhelp, time);
vorticity = curl_2D(vhelp, flow_field);
vorticity_maxima = local_maxima_bool_2(vhelp, vorticity, 101, treshold);
[span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, vorticity_maxima);

% Figure Parameters
filename = 'kp.jpg';
h = figure;
width=1000;
height=800;
set(gcf,'position',[10,10,width,height])
offset = 0.05;
%% draw on figure 

%draw_vorticity_maxima2D(vhelp, abs(vorticity), vorticity_maxima, h);
%draw_poincare_sections(vhelp, abs(vorticity), span_tree, pointsX, pointsY, h)
        
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


%%
title('Absolute Vorticity');
axis([0-offset 1+offset 0-offset 1+offset]);
drawnow 
    
% Capture the plot as an image 
frame = getframe(h); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 

imwrite(imind,cm,filename,'jpg'); 
