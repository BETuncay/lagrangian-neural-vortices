% test the ellipse fitting
clc;
clear;
close all;

path = 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results10\lcs_0025_3_2';
load(path);
domain = [0, 1; 0, 1];
resolution = [512, 512];
timespan = [3, 5];

ellipticColor = [0,.6,0];
ellipticColor2 = [.6,0,0];

[hfigure, hAxes] = setup_figure(domain);
title(hAxes,'elliptic LCSs')
xlabel(hAxes,'x achse')
ylabel(hAxes,'y achse')

% Plot finite-time Lyapunov exponent
cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
ftle_ = ftle(cgEigenvalue2,diff(timespan));
plot_ftle(hAxes,domain,resolution,ftle_);
colormap(hAxes,gray)

% Plot closed lambda lines
hClosedLambdaLinePos = plot_closed_orbit(hAxes,closedLambdaLinePos);
hClosedLambdaLineNeg = plot_closed_orbit(hAxes,closedLambdaLineNeg);
hClosedLambdaLine = [hClosedLambdaLinePos,hClosedLambdaLineNeg];
set(hClosedLambdaLine,'color',ellipticColor2)
drawnow

ellipticLcs = [elliptic_lcs(closedLambdaLinePos), elliptic_lcs(closedLambdaLineNeg)];

[hfigure, hAxes] = setup_figure(domain);
title(hAxes,'elliptic LCSs')
xlabel(hAxes,'x achse')
ylabel(hAxes,'y achse')

% Plot finite-time Lyapunov exponent
cgEigenvalue2 = reshape(cgEigenvalue(:,2),fliplr(resolution));
ftle_ = ftle(cgEigenvalue2,diff(timespan));
plot_ftle(hAxes,domain,resolution,ftle_);
colormap(hAxes,gray)

% Plot elliptic LCSs
hEllipticLcs = plot_elliptic_lcs(hAxes,ellipticLcs);
set(hEllipticLcs,'color',ellipticColor)
set(hEllipticLcs,'linewidth',2)


lcs = ellipticLcs{1};
a = fit_ellipse(lcs(:,1), lcs(:,2));


X = [lcs(:,1).*lcs(:,1), lcs(:,1).*lcs(:,2), lcs(:,2).*lcs(:,2), lcs(:,1), lcs(:,2), ones(size(lcs(:,1)))];


residue = X * a;
figure
imagesc([0, 1], [0, 1], Z);
set(gca,'YDir','normal') 
colorbar;



function a = fit_ellipse(x, y)

% Build desing matrix
D = [x.*x, x.*y, y.*y, x, y, ones(size(x))];
% Build scatter matrix
S = transpose(D) * D;
% Build 6x6 constraint matrix
C(6, 6) = 0; C(1,3) = -2; C(2,2) = 1; C(3,1) = -2;
% Solve generalised eigensystem
[gevec, geval] = eig(S, C);
%Find the only negative eigenvalue
[NegR, NegC] = find(geval < 0 & ~isinf(geval));
% Get fitted parameters
a = gevec(:, NegC);

end