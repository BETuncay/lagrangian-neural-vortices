% calculate flow map gradient of a flow map
function [gradF11,gradF12,gradF21,gradF22] = get_flowmap_gradient(Flow_Map)
    
    initialPositionX = squeeze(Flow_Map(1,:,1));
    initialPositionY = squeeze(Flow_Map(:,1,2));
    finalX = squeeze(Flow_Map(:,:,3));
    finalY = squeeze(Flow_Map(:,:,4));
    [gradF11,gradF12] = gradient(finalX,initialPositionX,initialPositionY);
    [gradF21,gradF22] = gradient(finalY,initialPositionX,initialPositionY); 




