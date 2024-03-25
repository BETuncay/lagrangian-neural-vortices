
% plots a flowmap
function plot_eta_field(vhelp, cgEigenvector, cgEigenvalue, lambda, step, fig, plot_title)
arguments
    vhelp vortex_helper
    cgEigenvector {mustBeNumeric}
    cgEigenvalue {mustBeNumeric}
    lambda {mustBeNumeric} = 1
    step  {mustBeNumeric} = 50
    fig matlab.ui.Figure = figure
    plot_title = 'Eta Field'
    
end

         
figure(fig);

[etaPos,~] = lambda_line(cgEigenvector,cgEigenvalue,lambda,'forceComplexNaN',true);
etaPos = reshape(etaPos,[get_sampled_resolution(vhelp, 1), get_sampled_resolution(vhelp, 2), 2]);

grid = get_mesh_grid2D(vhelp);
quiver(grid.x(1:step:end, 1:step:end), grid.y(1:step:end, 1:step:end), ...
        etaPos(1:step:end,1:step:end,1), etaPos(1:step:end,1:step:end,2));


title(plot_title);
axis([0,1 0,1])
end