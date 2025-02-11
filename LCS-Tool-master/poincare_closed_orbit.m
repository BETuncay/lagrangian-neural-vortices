% poincare_closed_orbit Find closed orbits using Poincare section map
%
% SYNTAX
% [closedOrbitPosition,orbitPosition] = poincare_closed_orbit(domain,resolution,vectorField,poincareSection)
% [closedOrbitPosition,orbitPosition] = poincare_closed_orbit(...,'odeSolverOptions',options)
% [closedOrbitPosition,orbitPosition] = poincare_closed_orbit(...,'nBisection',n)
% [closedOrbitPosition,orbitPosition] = poincare_closed_orbit(...,'dThresh',dThresh)
% [closedOrbitPosition,orbitPosition,hFigure] = poincare_closed_orbit(...,'showGraph',showGraph)
%
% INPUT ARGUMENTS
% showgraph: logical variable, set true to show plots of Poincare sections

function [closedOrbitPosition,orbitPosition,varargout] = poincare_closed_orbit(domain,resolution,vectorField,poincareSection,varargin)

p = inputParser;

addRequired(p,'domain',@(input)validateattributes(input,{'double'},{'size',[2,2],'real','finite'}))
addRequired(p,'resolution',@(input)validateattributes(input,{'double'},{'size',[1,2],'real','finite'}))
% FIXME Add validationFcn to addRequired
addRequired(p,'vectorField')
addRequired(p,'poincareSection')
addParameter(p,'odeSolverOptions',odeset)
addParameter(p,'nBisection',5,@(input)validateattributes(input,{'numeric'},{'scalar','>=',1,'integer'}));
addParameter(p,'dThresh',1e-2,@(input)validateattributes(input,{'double'},{'scalar','positive'}));
addParameter(p,'periodicBc',[false,false],@(input)validateattributes(input,{'logical'},{'size',[1,2]}));
addParameter(p,'showGraph',false,@(input)validateattributes(input,{'logical'},{'scalar'}))
addParameter(p,'checkDiscontinuity',true,@(input)validateattributes(input,{'logical'},{'scalar'}))

parse(p,domain,resolution,vectorField,poincareSection,varargin{:})

odeSolverOptions = p.Results.odeSolverOptions;
nBisection = p.Results.nBisection;
dThresh = p.Results.dThresh;
%periodicBc = [false,false];
periodicBc = p.Results.periodicBc;
showGraph = p.Results.showGraph;
checkDiscontinuity = p.Results.checkDiscontinuity;

% Poincare section vector
p = poincareSection.endPosition(2,:) - poincareSection.endPosition(1,:);

% Initial positions for Poincare orbits
orbitInitialPositionX = linspace(poincareSection.endPosition(1,1),poincareSection.endPosition(2,1),poincareSection.numPoints);
orbitInitialPositionY = linspace(poincareSection.endPosition(1,2),poincareSection.endPosition(2,2),poincareSection.numPoints);
orbitInitialPosition = transpose([orbitInitialPositionX;orbitInitialPositionY]);

orbitPosition = cell(poincareSection.numPoints,1);


% berkan
%x = domain(1,1):(domain(1,2) - domain(1,1))/ (resolution(1) - 1):domain(1,2);
%y = domain(2,1):(domain(2,2) - domain(2,1))/ (resolution(2) - 1):domain(2,2);
%[x, y] = meshgrid(x,y);
%ep = reshape(vectorField,[resolution(1),resolution(2), 2]);
%figure;
%Position = [300 10 1100 1100];
%quiver(x, y, ep(:,:,1),ep(:,:,2));
%hold on
%axis equal
% berkan

% integrate orbits
integrationLength = poincareSection.integrationLength;
endPosition = poincareSection.endPosition;

parfor idx = 1:poincareSection.numPoints
    %fprintf("Point number " + idx + " calculated in: ");
    orbitPosition{idx} = integrate_line(integrationLength,...
        orbitInitialPosition(idx,:),domain,resolution,periodicBc,...
        vectorField,odeSolverOptions,endPosition,...
        'checkDiscontinuity',checkDiscontinuity);
end


% berkan
%for r = 1:size(orbitPosition, 1)
%   orbit = orbitPosition{r};
%   plot(orbit(:,1), orbit(:,2));
%end
%drawnow
%berkan

% final position of orbits
orbitFinalPosition = cellfun(@(position)position(end,:),orbitPosition,'UniformOutput',false);
orbitFinalPosition = cell2mat(orbitFinalPosition);

xLength = hypot(diff(poincareSection.endPosition(:,1)),diff(poincareSection.endPosition(:,2)));

% angle of Poincare section with x-axis
theta = -atan((poincareSection.endPosition(1,2) - poincareSection.endPosition(2,2))/(poincareSection.endPosition(1,1) - poincareSection.endPosition(2,1)));
rotationMatrix = [cos(theta),-sin(theta);sin(theta),cos(theta)];

% Translate to origin
s(:,1) = orbitInitialPosition(:,1) - poincareSection.endPosition(1,1);
s(:,2) = orbitInitialPosition(:,2) - poincareSection.endPosition(1,2);

t(:,1) = orbitFinalPosition(:,1) - poincareSection.endPosition(1,1);
t(:,2) = orbitFinalPosition(:,2) - poincareSection.endPosition(1,2);

% Rotate to 0
s = rotationMatrix*transpose(s);
t = rotationMatrix*transpose(t);
s = transpose(s);
t = transpose(t);


if showGraph
    hFigure = figure;
    varargout{1} = hFigure;
    hAxes = axes;
    set(hAxes,'parent',hFigure)
    set(hAxes,'nextplot','add')
    title(hAxes,'Poincare map')
    set(hAxes,'box','on')
    set(hAxes,'xgrid','on')
    set(hAxes,'ygrid','on')
    set(hAxes,'xlim',[0,xLength])
    hReturnMap = plot(hAxes,abs(s(:,1)),t(:,1)-s(:,1));
    set(hReturnMap,'color','b')
    xlabel(hAxes,'s')
    ylabel(hAxes,'p(s) - s')
end

% find zero crossings of Poincare map (linear interpolation)
[~,closedOrbitInitialPosition] = crossing(t(:,1) - s(:,1),s(:,1));

if isempty(closedOrbitInitialPosition)
    closedOrbitPosition{1} = [NaN,NaN];
else
    if showGraph
        hClosedOrbitInitialPosition = plot(hAxes,abs(closedOrbitInitialPosition),zeros(size(closedOrbitInitialPosition)));
        set(hClosedOrbitInitialPosition,'LineStyle','none')
        set(hClosedOrbitInitialPosition,'marker','o')
        set(hClosedOrbitInitialPosition,'MarkerEdgeColor','b')
        set(hClosedOrbitInitialPosition,'DisplayName','Zero crossing candidate')
        hLegend = legend(hAxes,hClosedOrbitInitialPosition);
        set(hLegend,'location','best')
        drawnow
    end
    
    % Rotate to theta
    xx = [transpose(closedOrbitInitialPosition),zeros(numel(closedOrbitInitialPosition),1)];
    xx = rotationMatrix\transpose(xx);
    xx = transpose(xx);
    
    % Translate from origin
    closedOrbitInitialPositionX = xx(:,1) + poincareSection.endPosition(1,1);
    closedOrbitInitialPositionY = xx(:,2) + poincareSection.endPosition(1,2);
    
    closedOrbitInitialPosition = [closedOrbitInitialPositionX,closedOrbitInitialPositionY];
    
    % FILTER:
    % Discard discontinuous zero crossings
    % Refine zero crossings with bisection method
    % PARAMETERS
    distThresh = dThresh * xLength;
    
    nZeroCrossing = size(closedOrbitInitialPosition,1);
    for i = 1:nZeroCrossing
        
        neighboursOKFlag = 1;
        
        % find 2 neighbor points of zero crossing
        % FIXME Check if sorting by orbitInitialPosition(:,1) position
        % works if set Poincare section to vertical line
        [orbitInitialPositionSorted,ix] = sort(orbitInitialPosition(:,1));
        indx10 = find(closedOrbitInitialPosition(i,1) > orbitInitialPositionSorted,1,'last');
        indx20 = find(closedOrbitInitialPosition(i,1) < orbitInitialPositionSorted,1,'first');
        indx1 = min(ix(indx10),ix(indx20));
        indx2 = max(ix(indx10),ix(indx20));
        % neighbor points NOT clearly identified
        if any([isempty(indx1),isempty(indx2)]) || indx2 <= indx1 || abs(indx1-indx2) ~=1
            warning([mfilename,':selectNeighborOrbit'],'Selection of neighbor orbits failed.')
            closedOrbitInitialPosition(i,:) = NaN;
            % neighbor points clearly identified
        else
            % Bisection method
            % neighbor points
            p1 = orbitInitialPosition(indx1,:);
            p2 = orbitInitialPosition(indx2,:);
            
            for j = 1:nBisection
                % get return distance for p1, p2
                [p1finalPos,iEvent1] = integrate_line(poincareSection.integrationLength,p1,domain,resolution,periodicBc,vectorField,odeSolverOptions,poincareSection.endPosition,'checkDiscontinuity',checkDiscontinuity);
                if iEvent1 ~= 1
                    warning([mfilename,':bisectionOpenOrbit'],'open orbit in bisection method')
                end
                p1end = p1finalPos(end,:);
                p1dist = dot(p1end - p1,p/norm(p));
                [p2finalPos,iEvent2] = integrate_line(poincareSection.integrationLength,p2,domain,resolution,periodicBc,vectorField,odeSolverOptions,poincareSection.endPosition,'checkDiscontinuity',checkDiscontinuity);
                if iEvent2 ~= 1
                    warning([mfilename,':bisectionOpenOrbit'],'open orbit in bisection method')
                end
                p2end = p2finalPos(end,:);
                p2dist = dot(p2end - p2,p/norm(p));
                
                % in first iteration neighbour points must return to
                % Poincare section
                if any([iEvent1,iEvent2]~=1) && j==1
                    neighboursOKFlag = 0;
                    break
                end
                
                % Set integration length
                dPosition = diff(p2finalPos);
                length = sum(arrayfun(@(m)norm(dPosition(m,:)),1:size(dPosition,1)-1));
                
                % bisect
                p3 = (p1+p2)/2;
                % return distance for p3
                [p3finalPos,iEvent3] = integrate_line([0,1.1*length],p3,domain,resolution,periodicBc,vectorField,odeSolverOptions,poincareSection.endPosition,'checkDiscontinuity',checkDiscontinuity);
                if iEvent3 ~= 1
                    warning([mfilename,':bisectionOpenOrbit'],'open orbit in bisection method')
                end
                p3end = p3finalPos(end,:);
                p3dist = dot(p3end - p3,p/norm(p));
                
                if j ~= nBisection
                    if p1dist*p3dist < 0
                        p2 = p3;
                    else
                        p1 = p3;
                    end
                end
            end
            
            % linearly interpolated zero crossing point between final p1 and p2
            p4 = abs(p2dist)/(abs(p1dist)+abs(p2dist))*p1 + abs(p1dist)/(abs(p1dist)+abs(p2dist))*p2;
            
            % neighbor points of zero crossing must have a small return distance
            if any(abs([p1dist,p2dist]) > distThresh)
                closedOrbitInitialPosition(i,:) = NaN;                
            elseif neighboursOKFlag==0
                closedOrbitInitialPosition(i,:) = NaN;
            else
                closedOrbitInitialPosition(i,:) = p4;
            end
        end
    end
    % Erase invalid closed orbits
    [iy,~] = find(isnan(closedOrbitInitialPosition));
    closedOrbitInitialPosition(unique(iy),:) = [];
    
    if ~isempty(closedOrbitInitialPosition)
        nClosedOrbit = size(closedOrbitInitialPosition,1);
        % Integrate closed orbits
        closedOrbitPosition = cell(nClosedOrbit,1);
        for idx = 1:nClosedOrbit
            closedOrbitPosition{idx} = integrate_line(poincareSection.integrationLength,closedOrbitInitialPosition(idx,:),domain,resolution,periodicBc,vectorField,odeSolverOptions,poincareSection.endPosition,'checkDiscontinuity',checkDiscontinuity);
        end
        
        % FILTER: select outermost closed orbit
        s1(:,1) = closedOrbitInitialPosition(:,1) - poincareSection.endPosition(1,1);
        s1(:,2) = closedOrbitInitialPosition(:,2) - poincareSection.endPosition(1,2);
        distR = hypot(s1(:,1),s1(:,2));
        
        % Plot all valid zero crossings
        if showGraph
            delete(hLegend)
            delete(hClosedOrbitInitialPosition)
            hZeroCrossing = plot(hAxes,distR,zeros(size(distR)));
            set(hZeroCrossing,'LineStyle','none')
            set(hZeroCrossing,'marker','o')
            set(hZeroCrossing,'MarkerEdgeColor','b')
            set(hZeroCrossing,'MarkerFaceColor','b')
            set(hZeroCrossing,'DisplayName','Zero crossing')
            hLegend = legend(hAxes,hZeroCrossing);
            set(hLegend,'location','best')
            drawnow
        end
        
        % Sort closed orbits: closedOrbitPosition{1} is innermost,
        % closedOrbitPosition{end} is outermost.
        [~,sortIndex] = sort(distR);
        closedOrbitPosition = closedOrbitPosition(sortIndex);
        
        
        % berkan elliptic lcs must include poincare start point
        %sortIndexInside = [];
        %for i = 1:numel(sortIndex)
        %    orb = closedOrbitPosition{i};
        %    in = inpolygon(poincareSection.endPosition(1,1),poincareSection.endPosition(1,2), ...
        %                    orb(:,1),orb(:,2));
        %    if in
        %        sortIndexInside = [sortIndexInside; sortIndex(i)];
                %plot(orb(:,1), orb(:,2), 'r');
            %else
                %plot(orb(:,1), orb(:,2), 'b');
        %    end
        %end
        %
        %if isempty(sortIndexInside)
        %    closedOrbitPosition = cell(1,1);
        %    closedOrbitPosition{1} = [NaN,NaN];
        %else
        %    closedOrbitPosition = closedOrbitPosition(sortIndexInside);
        %end

        
        % Plot outermost zero crossing
        if showGraph
            hOutermostZeroCrossing = plot(hAxes,distR(sortIndex(end)),0);
            set(hOutermostZeroCrossing,'LineStyle','none')
            set(hOutermostZeroCrossing,'marker','o')
            set(hOutermostZeroCrossing,'MarkerEdgeColor','r')
            set(hOutermostZeroCrossing,'MarkerFaceColor','r')
            set(hOutermostZeroCrossing,'DisplayName','Outermost zero crossing')
            hLegend = legend(hAxes,[hZeroCrossing;hOutermostZeroCrossing]);
            set(hLegend,'Location','best')
            drawnow
        end
        
    else
        closedOrbitPosition{1} = [NaN,NaN];
    end
end

function [ind,t0,s0] = crossing(S,t,level,imeth)
% CROSSING find the crossings of a given level of a signal
%   ind = CROSSING(S) returns an index vector ind, the signal
%   S crosses zero at ind or at between ind and ind+1
%   [ind,t0] = CROSSING(S,t) additionally returns a time
%   vector t0 of the zero crossings of the signal S. The crossing
%   times are linearly interpolated between the given times t
%   [ind,t0] = CROSSING(S,t,level) returns the crossings of the
%   given level instead of the zero crossings
%   ind = CROSSING(S,[],level) as above but without time interpolation
%   [ind,t0] = CROSSING(S,t,level,par) allows additional parameters
%   par = {'none'|'linear'}.
%	With interpolation turned off (par = 'none') this function always
%	returns the value left of the zero (the data point thats nearest
%   to the zero AND smaller than the zero crossing).
%
%	[ind,t0,s0] = ... also returns the data vector corresponding to
%	the t0 values.
%
%	[ind,t0,s0,t0close,s0close] additionally returns the data points
%	closest to a zero crossing in the arrays t0close and s0close.
%
%	This version has been revised incorporating the good and valuable
%	bugfixes given by users on Matlabcentral. Special thanks to
%	Howard Fishman, Christian Rothleitner, Jonathan Kellogg, and
%	Zach Lewis for their input.

% Steffen Brueckner, 2002-09-25
% Steffen Brueckner, 2007-08-27		revised version

% Copyright (c) Steffen Brueckner, 2002-2007
% brueckner@sbrs.net

% Source: http://www.mathworks.com/matlabcentral/fileexchange/2432

% check the number of input arguments
narginchk(1,4);

% check the time vector input for consistency
if nargin < 2 || isempty(t)
    % if no time vector is given, use the index vector as time
    t = 1:length(S);
elseif length(t) ~= length(S)
    % if S and t are not of the same length, throw an error
    error('t and S must be of identical length!');
end

% check the level input
if nargin < 3
    % set standard value 0, if level is not given
    level = 0;
end

% check interpolation method input
if nargin < 4
    imeth = 'linear';
end

% make row vectors
t = t(:)';
S = S(:)';

% always search for zeros. So if we want the crossing of
% any other threshold value "level", we subtract it from
% the values and search for zeros.
S   = S - level;

% first look for exact zeros
ind0 = find( S == 0 );

% then look for zero crossings between data points
S1 = S(1:end-1) .* S(2:end);
ind1 = find( S1 < 0 );

% bring exact zeros and "in-between" zeros together
ind = sort([ind0 ind1]);

% and pick the associated time values
t0 = t(ind);
s0 = S(ind);

if strcmp(imeth,'linear')
    % linear interpolation of crossing
    for ii=1:length(t0)
        if abs(S(ind(ii))) > eps(S(ind(ii)))
            % interpolate only when data point is not already zero
            NUM = (t(ind(ii)+1) - t(ind(ii)));
            DEN = (S(ind(ii)+1) - S(ind(ii)));
            DELTA =  NUM / DEN;
            t0(ii) = t0(ii) - S(ind(ii)) * DELTA;
            % I'm a bad person, so I simply set the value to zero
            % instead of calculating the perfect number ;)
            s0(ii) = 0;
        end
    end
end

% FIXME Causes error when minimum is between first and second point
% % Addition:
% % Some people like to get the data points closest to the zero crossing,
% % so we return these as well
% [~,II] = min(abs([S(ind-1) ; S(ind) ; S(ind+1)]),[],1);
% ind2 = ind + (II-2); %update indices
%
% t0close = t(ind2);
% s0close = S(ind2);