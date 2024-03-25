% elliptic_lcs Given closed lambda lines, return elliptic LCSs
%
% SYNTAX
% ellipticLcs = elliptic_lcs(closedLambdaLine)
%
% INPUT ARGUMENTS
% closedLambdaLine: closed lambda line positions returned from
% discard_empty_closed_lambda
%
% OUTPUT ARGUMENT
% ellipticLcs: elliptic LCS position for each poincare section over a range
% of lambda values. Cell array with number of elements equal to number of
% Poincare sections
%
% EXAMPLES
% x positions of elliptic LCS for third poincare section:
% [ellipticLcs] = elliptic_lcs(closedLambdaLine)
% ellipticLcs{3}(:,1)
% 
% y positions of elliptic LCS for fourth poincare section:
% ellipticLcs{4}(:,2)
%
% number of elliptic LCSs:
% sum(~cellfun(@isempty,ellipticLcs))
%
% lambda value of closed orbit of second Poincare section:
% lambda(outerPosI(2))

function ellipticLcs = elliptic_lcs(closedLambdaLine, pointsX, pointsY, method, periodic, domain, area_weight, ellipse_threshold)
arguments
    closedLambdaLine
    pointsX
    pointsY
    method
    periodic = [true, true]
    domain = [0, 1; 0, 1]
    area_weight = 0.000125
    ellipse_threshold = 1e-5
end

nLambda = size(closedLambdaLine,1);
nPoincareSection = size(closedLambdaLine,2);
nPoints = size(pointsX,1);

orbits_per_lcs = cell(1, nPoints);
orbits_without_lcs = cell(1, 1);


% simplify and reorganise datastructure
% --> list of points for each poincare section

% x component periodic
if periodic(1)
    pointsX = [pointsX; pointsX - (domain(1,2) - domain(1,1)); pointsX + (domain(1,2) - domain(1,1))];
    pointsY = [pointsY; pointsY; pointsY];
end

% y component periodic
if periodic(2)
    pointsY = [pointsY; pointsY - (domain(2,2) - domain(2,1)); pointsY + (domain(2,2) - domain(2,1))];
    pointsX = [pointsX; pointsX; pointsX];
end

for iPoincareSection = 1:nPoincareSection
    for iLambda = 1:nLambda
        for iOrbit = 1:numel(closedLambdaLine{iLambda,iPoincareSection})
            orbit = closedLambdaLine{iLambda,iPoincareSection}{iOrbit};
            in = inpolygon(pointsX, pointsY, orbit(:,1),orbit(:,2));
            if(any(in))
                idx = find(in);
                idx = mod(idx - 1, nPoints) + 1;
                for i = 1:size(idx, 1)
                    orbits_per_lcs{idx(i)}{end+1} = orbit;
                end
            else
                if ~any(isnan(orbit))
                    orbits_without_lcs{end+1} = orbit;
                end
            end
        end
    end
end

orbits_per_lcs = orbits_per_lcs(~cellfun(@isempty,orbits_per_lcs));
nPoincareSection = size(orbits_per_lcs, 2);
ellipticLcs = cell(1,size(orbits_per_lcs,1));

switch method
    case 'original' % deprecated, use the adjusted version
        for iPoincareSection = 1:nPoincareSection
            maxArea = nan;
            for iLambda = 1:nLambda
                outerClosedLambdaLine = closedLambdaLine{iLambda,iPoincareSection}{end};
                iArea = polyarea(outerClosedLambdaLine(:,1),outerClosedLambdaLine(:,2));
                if isnan(maxArea)
                    if ~isnan(iArea)
                        maxArea = iArea;
                        ellipticLcs{iPoincareSection} = outerClosedLambdaLine;
                    end
                else
                    if ~isnan(iArea)
                        if iArea > maxArea
                            maxArea = iArea;
                            ellipticLcs{iPoincareSection} = outerClosedLambdaLine;
                        end
                    end
                end
            end
        end
    case 'original_adjusted'
        for iPoincareSection = 1:nPoincareSection
            maxArea = -inf;
            for iOrbit = 1:size(orbits_per_lcs{iPoincareSection},2)
                
                outerClosedLambdaLine = orbits_per_lcs{iPoincareSection}{iOrbit};
                iArea = polyarea(outerClosedLambdaLine(:,1),outerClosedLambdaLine(:,2));
                
                if iArea > maxArea
                    maxArea = iArea;
                    ellipticLcs{iPoincareSection} = outerClosedLambdaLine;
                end
            end
        end
    case 'ellipse_metric'
        for iPoincareSection = 1:nPoincareSection
            minDifference = inf;
            for iOrbit = 1:size(orbits_per_lcs{iPoincareSection},2)
                
                mostElliptic_LambdaLine = orbits_per_lcs{iPoincareSection}{iOrbit};
                residue = calculate_residue(mostElliptic_LambdaLine);
                difference = calculate_ellipse_difference(mostElliptic_LambdaLine, residue);
                
                if difference < minDifference
                    minDifference = difference;
                    ellipticLcs{iPoincareSection} = mostElliptic_LambdaLine;
                end
            end
        end
    case 'ellipse_area_metric'
        for iPoincareSection = 1:nPoincareSection
            minDifference = inf;
            for iOrbit = 1:size(orbits_per_lcs{iPoincareSection},2)
                
                mostElliptic_LambdaLine = orbits_per_lcs{iPoincareSection}{iOrbit};
                residue = calculate_residue(mostElliptic_LambdaLine);
                difference = calculate_ellipse_difference(mostElliptic_LambdaLine, residue);
                iArea = polyarea(mostElliptic_LambdaLine(:,1),mostElliptic_LambdaLine(:,2));
                difference = (abs(difference)) / (iArea);
                if difference < minDifference
                    minDifference = difference;
                    ellipticLcs{iPoincareSection} = mostElliptic_LambdaLine;
                end
            end
        end
        % imagine the ellipse difference and area to be a vector
        % [diff; area] ==> find vector with maximal angle to [1;0] 
    case 'ellipse_area_min_angle'
        v = [1;0];
        for iPoincareSection = 1:nPoincareSection
            maxAngle = -inf;
            for iOrbit = 1:size(orbits_per_lcs{iPoincareSection},2)
                
                mostElliptic_LambdaLine = orbits_per_lcs{iPoincareSection}{iOrbit};
                residue = calculate_residue(mostElliptic_LambdaLine);
                difference = calculate_ellipse_difference(mostElliptic_LambdaLine, residue);
                iArea = polyarea(mostElliptic_LambdaLine(:,1),mostElliptic_LambdaLine(:,2));
                u = [abs(difference); iArea^2];
                angle = acos(min(1, max(-1, u(:) .' * v(:) / norm(u))));
                if angle > maxAngle
                    maxAngle = angle;
                    ellipticLcs{iPoincareSection} = mostElliptic_LambdaLine;
                end
            end
        end
    case 'ellipse_area_linear_combination'     
        for iPoincareSection = 1:nPoincareSection
            minValue = inf;
            for iOrbit = 1:size(orbits_per_lcs{iPoincareSection},2)
                
                mostElliptic_LambdaLine = orbits_per_lcs{iPoincareSection}{iOrbit};
                residue = calculate_residue(mostElliptic_LambdaLine);
                difference = calculate_ellipse_difference(mostElliptic_LambdaLine, residue);
                if difference > ellipse_threshold
                   disp('ist das noch eine ellipse ?')
                   disp(difference);
                   continue
                end
                iArea = polyarea(mostElliptic_LambdaLine(:,1),mostElliptic_LambdaLine(:,2));
                value = difference - area_weight * iArea;
                if value < minValue
                    %disp(value);
                    minValue = value;
                    ellipticLcs{iPoincareSection} = mostElliptic_LambdaLine;
                end
            end
        end
       % linear weighted sum
    otherwise
        disp('other value')
end



% Discard ellipticLcs elements for Poincare sections without closed orbits
ellipticLcs = ellipticLcs(~cellfun(@isempty,ellipticLcs));

end

% deprecated
function [orbit_index, min_difference] = get_minimunm_difference(lcs_list)
    min_difference = inf;
    orbit_index = 0;
    for i = 1:numel(lcs_list) 
        if all(~isnan(lcs_list{i}))
            residue = calculate_residue(lcs_list{i});
            difference = calculate_ellipse_difference(lcs_list{i}, residue);
            
            if difference < min_difference
                min_difference = difference;
                orbit_index = i;
            end
        end
    end
end

% Calculates the weighted difference of an orbit to its fitted ellipse
% how it works:
% example: we have points: 1, 2, 3 with edges: a, b, c
% 1 -a- 2
% |    /
% c   b
% |  /
%  3  
%
% 1. we want to calculate all connecting vectors between our orbit points
% let be lcs = [1; 2; 3] => padded_lcs = [3; 1; 2; 3; 1]
% --> [1; 2; 3; 1] - [3; 1; 2; 3] = [1 - 3; 2 - 1; 3 - 2; 1 - 3] = [c, a, b, c]
% ==> padded_lcs(2:end, :) - padded_lcs(1:end - 1, :);
% 2. calculate the length of each vector
% ==> hypot(weights(:,1), weights(:,2));
% 3. calculate the weight of each point: e.g. weight of 1 == len(a) + len(c)
% ==> movsum(weights,2, 'Endpoints', 'discard');
% 4. calculate the weighted difference
% ==> dot(weights, residue) / sum(weights);

function difference = calculate_ellipse_difference(lcs, residue)
    
    %residue = algebraic distance, fitting by minimizing the sum of squared algebraic distances
    residue = residue .* residue;
    padded_lcs = padarray(lcs, [1],'circular');
    weights = padded_lcs(2:end, :) - padded_lcs(1:end - 1, :);
    weights = hypot(weights(:,1), weights(:,2));
    weights = movsum(weights,2, 'Endpoints', 'discard');
    difference = dot(weights, residue) / sum(weights);
end


% calculate residue = difference of orbit to calculated ellipse
% X * a == distance field of ellipse
function residue = calculate_residue(lcs)

    a = fit_ellipse(lcs(:,1), lcs(:,2));
    X = [lcs(:,1).*lcs(:,1), lcs(:,1).*lcs(:,2), lcs(:,2).*lcs(:,2), lcs(:,1), lcs(:,2), ones(size(lcs(:,1)))];     
    residue = X * a;
end


% fit ellipse to input points
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