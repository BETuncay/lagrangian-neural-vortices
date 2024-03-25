%% Cleanup
clear;
close all;
clc;

%% Parameters
%pathIn = 'D:\_Tools_Data\Matlab_Data\Vortices Dataset\1000.am';
pathIn = 'G:\Fundue\3000.am'; % von 0000 bis 7995, jedes 5te element
sampling_rate = 1;%20;
sampling_rate_time = 1;

%% Code
vhelp = vortex_helper(pathIn, sampling_rate);
grid = get_mesh_grid2D(vhelp);


flow_field.u = squeeze(vhelp.data(2, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:sampling_rate_time:vhelp.res(3)));
flow_field.v = squeeze(vhelp.data(1, 1:vhelp.sampling_rate:vhelp.res(1), 1:vhelp.sampling_rate:vhelp.res(2), 1:sampling_rate_time:vhelp.res(3)));

%flow_field = get_flow_vector_field3D(vhelp);
vorticity = curl_2D(vhelp, flow_field);

h = figure;
width=1000;
height=800;
set(gcf,'position',[10,10,width,height])

offset = 0.05;
filename = 'abs_vorticity.gif';
DelayTime = 0;
for n = 1:size(flow_field.u, 3)
    
    %quiver(grid.x, grid.y, flow_field.u(:,:,n), flow_field.v(:,:,n), 1);
    draw_vorticity2D(vhelp, abs(vorticity(:,:,n)), h);
    title('Absolute Vorticity');
    axis([0-offset 1+offset 0-offset 1+offset]);
    drawnow 
    
    % Capture the plot as an image 
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if n == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf, 'DelayTime', DelayTime); 
    else 
        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', DelayTime); 
    end 
end