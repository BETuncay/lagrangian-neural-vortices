clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean';
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);
plot_results = true;


for i = 3:size(listing, 1)
    result = listing(i).name;
    load(result);
    disp(result);
    
    old_lcs_mask = lcs_mask;
    lcs_mask = boolean_mask_list(vhelp, ellipticLcs, false);
    
    if plot_results
        f = figure;
        f.Position = [0 100 900 700];
        tiledlayout(1,2)
        nexttile
        draw_vorticity2D(vhelp, old_lcs_mask, f);
        
        nexttile
        draw_vorticity2D(vhelp, lcs_mask, f);
        drawnow
    end
    
    save(result, 'lcs_mask', '-append'); 
end