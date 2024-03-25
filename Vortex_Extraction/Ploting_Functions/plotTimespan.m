clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Timespan_Comparison';
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);

for i = 3:size(listing, 1)
    result = listing(i).name;
    load(result);
    
    sample_per_flow_field = numel(eliptic_lcs_list); % amount of elliptic lcs extracted from one flow field
    for j = 1:sample_per_flow_field
        
        % extract initial flow field
        initial_flow.u = squeeze(flow_field_list{j, 1}(1,:,:));
        initial_flow.v = squeeze(flow_field_list{j, 2}(1,:,:));
        % calculate vorticity, vorticity maxima and poincare sections
        vorticity = curl_2D(vhelp, initial_flow);
        if exist('voricity_maxima_list','var') == 0
            local_max_points = local_maxima_bool_2(vhelp, vorticity, 99);
        else
            local_max_points = voricity_maxima_list{j};
        end
        [span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, local_max_points);

        % plot vorticity with the poincare sections overlayed
        fig = figure;
        draw_poincare_sections(vhelp, vorticity, span_tree, pointsX, pointsY, fig)
        title(result, j);
        hold on
        
        % plot the calculated elliptic lcs
        for k = 1:size(eliptic_lcs_list{j}, 2)
            lcs = eliptic_lcs_list{j}{k};
            plot(lcs(:,1), lcs(:,2), 'color','g','linewidth',5);
        end
        drawnow
    end
end