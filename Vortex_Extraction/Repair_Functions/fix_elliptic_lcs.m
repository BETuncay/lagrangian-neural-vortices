clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold_clean';
current_path = pwd;
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);
ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];
plot_results = true;

if plot_results
    fig = figure;
end


for i = 3:size(listing, 1)
    result = listing(i).name;
    load(result);
    
    [span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, vorticity_maxima);
    old_ellipticLcs = ellipticLcs;
    
    % calculate new_lcs
    ellipticLcs = [elliptic_lcs(closedLambdaLinePos, pointsX, pointsY, 'ellipse_area_linear_combination'), elliptic_lcs(closedLambdaLineNeg, pointsX, pointsY, 'ellipse_area_linear_combination')];
    
    if plot_results
        % old elliptic lcs   
        fig.Position = [0 100 900 700];
        tiledlayout(1,2)
        ax1 = nexttile;
        hold on
        plot_lcs(vhelp, flow_field_u, flow_field_v, pointsX, pointsY, old_ellipticLcs, closedLambdaLinePos, closedLambdaLineNeg, ellipticColor, fig, ax1, 'Before')

        % new elliptic lcs   
        ax2 = nexttile;
        hold on
        plot_lcs(vhelp, flow_field_u, flow_field_v, pointsX, pointsY, ellipticLcs, closedLambdaLinePos, closedLambdaLineNeg, ellipticColor, fig, ax2, 'After')
    end
    
    save(result, 'ellipticLcs', 'closedLambdaLinePos', 'closedLambdaLineNeg', '-append'); 
end

cd(current_path);


function plot_lcs(vhelp, flow_field_u, flow_field_v, pointsX, pointsY, ellipticLcs, closedLambdaLinePos, closedLambdaLineNeg, ellipticColor, fig, ax, plot_title)

        flow_field.u = squeeze(flow_field_u(1,:,:));
        flow_field.v = squeeze(flow_field_v(1,:,:));
        vorticity = curl_2D(vhelp, flow_field);

        draw_vorticity2D(vhelp, abs(vorticity), fig, plot_title);
        hClosedLambdaLinePos = plot_closed_orbit(ax,closedLambdaLinePos);
        hClosedLambdaLineNeg = plot_closed_orbit(ax,closedLambdaLineNeg);
        set(hClosedLambdaLinePos, 'color',ellipticColor)
        set(hClosedLambdaLineNeg, 'color',ellipticColor)

        for k = 1:size(ellipticLcs, 2)
            lcs = ellipticLcs{k};
            plot(lcs(:,1), lcs(:,2), 'color','g','linewidth',5);
        end
        
        
        plot(pointsX, pointsY, 'r*');
end