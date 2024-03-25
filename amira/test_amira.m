clear all;

pathIn = 'D:\_Tools_Data\Matlab_Data\Vortices Dataset\2000.am';
pathOut = 'C:\Datasets\Qinzhu\SmokeBuoyancy\Frame10_test.am';

% write an Amira file
writeAmira(pathOut, data, domainMin, domainMax);

% read an Amira file
[data, domainMin, domainMax, res, numComponents] = readAmira(pathIn);
