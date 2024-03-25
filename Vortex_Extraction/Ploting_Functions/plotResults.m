clc
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Training_Data_Set';%quick test

current_path = pwd;
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
flow_map_detail = 30;

for i = 3:size(listing, 1)
    cd(result_path);
    result = listing(i).name;
    load(result);
    cd(current_path);
    close all
    time = str2double(time_str);
    interval = str2double(interval_str);
    timespan = [time, time+interval];
    flow_field.u = squeeze(flow_field_u(1,:,:));
    flow_field.v = squeeze(flow_field_v(1,:,:));
    vorticity = curl_2D(vhelp, flow_field);
    disp(result);
    
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
    %movegui(fig,'center');
    tiledlayout(2,2)
    sgtitle(result,'Interpreter','none');
    
    % plot abs vorticity with the poincare sections overlayed
    ax1 = nexttile;
    draw_poincare_sections(vhelp, abs(vorticity), span_tree, pointsX, pointsY, fig, 'Absolute Vorticity');
    
    % Plot Flow Map
    ax2 = nexttile;
    plot_flowmap(Flow_Map, flow_map_detail, fig, 'Flow Map');
    colormap(ax2,gray)
    
    % Plot FTLE, Psects and elliptic LCS
    ax3 = nexttile;
    cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
    ftle_ = ftle(cgEigenvalue2,diff(timespan));
    plot_ftle(ax3,domain,resolution,ftle_);
    colormap(ax3,gray)
    title(ax3,'elliptic LCSs')
    set(gca,'YDir','normal') 
    
    hold on
    % Plot Poincare sections
    for psec = 1:size(poincareSection,2)
        pos = poincareSection(psec).endPosition;
        plot(ax3,pos(:,1),pos(:,2), 'Color', ellipticColor, 'LineStyle','--','marker','o','MarkerFaceColor',ellipticColor,'MarkerEdgeColor','w');
    end

    % Plot closed lambda lines
    hClosedLambdaLinePos = plot_closed_orbit(ax3,closedLambdaLinePos);
    hClosedLambdaLineNeg = plot_closed_orbit(ax3,closedLambdaLineNeg);
    set(hClosedLambdaLinePos, 'color',ellipticColor)
    set(hClosedLambdaLineNeg, 'color',ellipticColor)

    % Plot elliptic LCSs
    hEllipticLcs = plot_elliptic_lcs(ax3,ellipticLcs);
    set(hEllipticLcs,'color',ellipticColor2)
    set(hEllipticLcs,'linewidth',2)
    
    % Plot Binary Mask
    ax4 = nexttile;
    draw_vorticity2D(vhelp, lcs_mask, fig, 'Binary Mask');
    colormap(ax4,gray)
    drawnow
end