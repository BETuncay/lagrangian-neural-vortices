Modern approaches to vortex detection usually require cost-intensive computations.
The goal of this paper is to design and train an U-Net which is capable of detecting objective vortices.
Thus, we shift the computation time to the network training and subsequently achieve fast vortex detection.

The fluid flow dataset consists of a wide spectrum of numerically simulated laminar and turbulent time-dependent 2D vector fields.
Source: https://cgl.ethz.ch/publications/papers/paperJak20a.php

The training dataset used to train the U-Net is extracted using functionalities of the LCS Tool developed by the Nonlinear Dynamical Systems Group at ETH Zurich, led by Prof. George Haller.
Source: https://github.com/jeixav/LCS-Tool

The U-Net is implemented in keras with a tensorflow backbone.

**Usage**

The two folders: _Vortex Extraction_ and _U-Net_ contain all relevant code. The Vortex Extraction folder contains all Matlab files involved in the extraction of vortices. The U-Net folder contains all python files required to create, train and test neural networks.


**Vortex Extraction:**

The folder "amira", "LCS-Tool-Master" and "Vortex Extraction" need to be added to the Matlab path. (Home -> Set Path -> Add folder)

The main file is "generate_training_data.m" and executing this file starts the vortex extraction process.
All relevant parameters must be set in the config struct.
This will call "extract_elliptic_lcs.m" in a loop which iterates through all specified fluid flows and extracts vortices / calculates their binary mask.
The result along with different input types are saved.

3 Subfolders are also contained, which serve various purposes.
- Plotting_Functions contains plotting functions. Here " plotResults.m" is very usefull to evalutate the ground truth results.
- Repair_Functions are used to repair erronous data in the results. Input types, vortex boundaries, binary masks can be recalculated here.
- Other_Functions consists of old and deprecated files.


**U-Net:**

- "train_network.py" is used to train new networks. The U-Net model, hyperparameters, input type and training data paths need to be set beforehand.
- "lcs_unet.py" contains the code used to build the U-Net model
- "get_training_data.py" is used to load the input and ground truth data for training.
- "compare_network.py" is used to plot the predictions of one or several networks.

