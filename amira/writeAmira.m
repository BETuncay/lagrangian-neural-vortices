function writeAmira(pathOut, data, domainMin, domainMax, res, numComponents)
%WRITEAMIRA Exports as an Amira file

% remove singleton dimensions
data = squeeze(data);

% if a vector field
if length(size(data)) == 4
    
    % read number of components if not provided
    if nargin < 6
        numComponents = int32(size(data,1));
    end

    % read grid resolution if not provided
    if nargin < 5
        resX = size(data,2);
        resY = size(data,3);
        resZ = size(data,4);
        res = int32([resX; resY; resZ]);
    end
    
else
    % if scalar field, then numComponents = 1
    numComponents = int32(1);
    % read grid resolution if not provided
    if nargin < 5
        resX = size(data,1);
        resY = size(data,2);
        resZ = size(data,3);
        res = int32([resX; resY; resZ]);
    end
end

% call the C code to write the file
cwrite_amira(pathOut, data, single(domainMin), single(domainMax), res, numComponents);

end

