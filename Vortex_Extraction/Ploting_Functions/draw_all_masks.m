clc;
clear;
close all;

result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean';
current_path = pwd;
cd(result_path);
listing = dir(result_path);
vhelp = vortex_helper('',1, false);

for i = 3:size(listing,1)
    cd(result_path);
    result = listing(i).name;
    load(result);
    cd(current_path);

    plot_mask(vhelp, lcs_mask, 'Boolean Mask');
    drawnow
end


function plot_mask(vhelp, matrix, name)
    figure
    imagesc([vhelp.domainMin(1), vhelp.domainMax(1)], [vhelp.domainMin(2), vhelp.domainMax(2)], matrix);
    title(name);
    set(gca,'YDir','normal');
    colorbar;
end