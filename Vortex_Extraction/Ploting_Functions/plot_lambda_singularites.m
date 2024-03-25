clc;
clear;
close all;
result_path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results15_low_threshold';
cd(result_path);
listing = dir(result_path);

vhelp = vortex_helper('',1, false);
ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];
plot_results = true;


% example 
for i = 3:size(listing, 1)
    result = listing(i).name;
    load(result);
    
    flow_field.u = squeeze(flow_field_u(1,:,:));
    flow_field.v = squeeze(flow_field_v(1,:,:));
    vorticity = curl_2D(vhelp, flow_field);
    [span_tree, edges, edgeSize, pointsX, pointsY] = get_poincare_section_graph(vhelp, vorticity_maxima);
    
    resolution = [512, 512];
    eps = 0.2;
    cgEigenvectors = reshape(cgEigenvector(:,:),[fliplr(resolution), 2, 2]);
    cgEigenvalues = reshape(cgEigenvalue(:,:),[fliplr(resolution), 2]);

   
    Singularities = false(512, 512);
    for j = 1:512
        for k = 1:512
                x0 = [j;k];
                E = general_Green_Lagrange(x0, cgEigenvalues, cgEigenvectors);
                r = rcond(E);
                if or(r < 0.001, isnan(r))
                    Singularities(j, k) = true;
                end

        end
    end
    
    fig = figure;
    ax1 = axis;
    fig.Position = [0 100 900 700];
    hold on
    draw_poincare_sections(vhelp, vorticity, span_tree, pointsX, pointsY, fig);
    
    hold on
    [pointsY, pointsX] = find(Singularities);
    pointsX = transform_to_DomainCoords(vhelp, pointsX, 1);
    pointsY = transform_to_DomainCoords(vhelp, pointsY, 2);
    plot(pointsX, pointsY, 'r*', 'color', 'r');

    hold on
    for k = 1:size(ellipticLcs, 2)
        lcs = ellipticLcs{k};
        plot(lcs(:,1), lcs(:,2), 'color','g','linewidth',5);
    end

     drawnow
end


function E = general_Green_Lagrange(x0, cg_values, cg_vector)

S = [cg_vector(x0(1), x0(2), 1, 1), cg_vector(x0(1), x0(2), 2, 1); ...
    cg_vector(x0(1), x0(2), 1, 2), cg_vector(x0(1), x0(2), 2, 2)];

L = [cg_values(x0(1), x0(2), 1), 0 ; 0 , cg_values(x0(1), x0(2), 2)];

C = S * L * inv(S);

I = [1,0;0,1];
%E = 0.5 .* (C - (L .* L));
E = 0.5 .* (C - I);

end