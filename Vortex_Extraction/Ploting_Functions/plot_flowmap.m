
% plots a flowmap
function plot_flowmap(flowmap, step, fig, plot_title)
arguments
    flowmap (:, :, 4) {mustBeNumeric}
    step {mustBeNumeric} = 50
    fig matlab.ui.Figure = figure
    plot_title = 'Flow Map'
    
end

figure(fig);
quiver(squeeze(flowmap(1:step:end,1:step:end,1)), squeeze(flowmap(1:step:end,1:step:end,2)), ...
        squeeze(flowmap(1:step:end,1:step:end,3) - flowmap(1:step:end,1:step:end,1)), ...
        squeeze(flowmap(1:step:end,1:step:end,4) - flowmap(1:step:end,1:step:end,2)));
    
    
title(plot_title);
axis([0,1 0,1])
end