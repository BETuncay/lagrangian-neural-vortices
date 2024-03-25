% similar to generate_training_data, but different intervals are calculated
warning('off','all')
clc;
close all;
clear;

if isempty(gcp('nocreate'))
    pool = parpool('threads');
end


config = struct('sampling_rate', 5, ... step size for loading the flow data \in [1, 512]
                'extraction_times', [4], ... times at with vortices are extracted \in [0, 10]
                'interval', 3,  ... extraction_times (= start time) + interval define the timespan for cgTensor calculation
                'plot_results', true, ... plot results, binary mask
                'save_data', true, ... should results be saved
                'save_folder_path', 'D:\_Tools_Data\Matlab_Data\lagrangian-neural-vortices\Vortex_Extraction\Results\Results3',...
                'poincareSection_numPoints', 250, ... amount of initial integration points per poincare section
                'orbitMaxLength', 4, ... max length of a lambda line
                'trajectory_numPoints', 20 ... amount of points to calculate flow trajectory at (U-Net input)
);

 pathIn = 550;

for interval = 0.01%:0.04:5
    extract_elliptic_lcs(pathIn, config);
end

% lcs 3400 with interval 0.1 good results




for pathIn = 5:5:2000
    disp(pathIn);
    try
        extract_elliptic_lcs(pathIn, config);
    catch ME
        fprintf('elliptic lcs extraction error: %s\n', ME.message);
        continue; 
    end
end

