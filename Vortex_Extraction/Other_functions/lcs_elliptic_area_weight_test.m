% function to test the weight parameter for picking votex boundaries
clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set_new';
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);
ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];

fig = figure;
fig.Position = [0 100 900 700];

method = 'ellipse_area_linear_combination';
periodic = [true, true];
domain = [0, 1; 0, 1];
ellipse_threshold = 1;


for i = 56:size(listing, 1)
    result = listing(i).name;
    load(result);
    flow_field.u = squeeze(flow_field_u(1,:,:));
    flow_field.v = squeeze(flow_field_v(1,:,:));
    vorticity = curl_2D(vhelp, flow_field);
    
    [pointsY, pointsX] = find(vorticity_maxima);
    pointsX = transform_to_DomainCoords(vhelp, pointsX, 1);
    pointsY = transform_to_DomainCoords(vhelp, pointsY, 2);

    tiledlayout(1,3)
    for weight = [1, 0.0002, 0.0007]
        
        
        ellipticLcs = [elliptic_lcs(closedLambdaLinePos, pointsX, pointsY, method, periodic, domain, weight, ellipse_threshold),...
                      elliptic_lcs(closedLambdaLineNeg, pointsX, pointsY, method, periodic, domain, weight, ellipse_threshold)];
        ax = nexttile;
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XColor', 'none','YColor','none')
        hold on
        if weight == 1
            ax_name = 'outermost curve';
        else
            ax_name = 'linear combination with \alpha = ' + string(weight); 
        end
        draw_vorticity2D(vhelp, abs(vorticity), fig, ax_name);
        axis equal
        hClosedLambdaLinePos = plot_closed_orbit(ax,closedLambdaLinePos);
        hClosedLambdaLineNeg = plot_closed_orbit(ax,closedLambdaLineNeg);
        set(hClosedLambdaLinePos, 'color',ellipticColor)
        set(hClosedLambdaLineNeg, 'color',ellipticColor)

        for k = 1:size(ellipticLcs, 2)
            lcs = ellipticLcs{k};
            plot(lcs(:,1), lcs(:,2), 'color',ellipticColor2,'linewidth', 3);
        end
        drawnow
    end
end