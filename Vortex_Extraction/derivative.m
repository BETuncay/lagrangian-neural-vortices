% derivative function adjusted from LCSTOOL
%
% SYNTAX
% derivative_ = derivative(time,position,VX_interpolant,VY_interpolant)
%
% INPUT ARGUMENTS
% time: scalar
% position: [x1;y1;x2;y2;...;xN;yN]
% VX_interpolant: griddedInterpolant for x-component of velocity
% VY_interpolant: griddedInterpolant for y-component of velocity
%
% OUTPUT ARGUMENT
% derivative_: [xVelocity1;yVelocity1;xVelocity2;yVelocity2;...;xVelocityN;yVelocityN]

function  derivative_ = derivative(time,position,VX_interpolant,VY_interpolant,periodicBc,domain)

nPosition = numel(position)/2;
derivative_ = nan(nPosition*2, 1, 'single');

for m = 1:numel(periodicBc)
    if periodicBc(m)
        position(m:2:end + m - 2) = mod(position(m:2:end + m - 2) - domain(m,1), diff(domain(m,:))) + domain(m,1);
    end
end
% time
time = time*ones(nPosition,1, 'single');

% x-positions
derivative_(1:2:end-1) = VX_interpolant(time,position(2:2:end),position(1:2:end-1));
% y-positions
derivative_(2:2:end) = VY_interpolant(time,position(2:2:end),position(1:2:end-1));

