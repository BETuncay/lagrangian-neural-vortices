function [data,domainMin,domainMax,res,numComponents] = readAmira(pathIn)
%READAM Reads an Amira vector field. Dimensions: numComponents x X x Y x Z

[data, domainMin, domainMax, res, numComponents] = cread_amira(pathIn);
if numComponents == 1
    data = reshape(data, res(1), res(2), res(3));
else
    data = reshape(data, numComponents, res(1), res(2), res(3));
end

% end of function
end

