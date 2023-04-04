# STASIS Visualization Workflow
Carina Fuss, 28.02.2023

## List of Figures
1. Image showing a vector of the worst field direction for power or temperature next to a smoothed and naturally colored implant (maybe multiple such images with the implant seen from different angles)
2. Plot showing the voxeled implant where each voxel is colored according to the worst possible temperature rise at that voxel
3. Plot showing the voxeled implant where each voxel is colored according to the temperature rise when the field points in the direction corresponding to the voxel with the highest temperature rise
4. Vector field plot showing the worst directions for some voxels (so that the plot is still visually appealing), with vectors that are colored according to the worst possible temperature rise at that voxel

## Workflow
1.	run script to set up simulations
2.	change thermal solver settings from CPU to GPU manually
3.	run thermal simulations
4.	run script to extract simulation results and set up visualizations
5.	for worst direction thermal sim: change thermal solver settings from CPU to GPU manually and run
6.	create the two thermal voxel figures using surface viewers, adapt scaling

### Create Vector Figures
7.	backup copy!
8.	change colors of implant and smooth surface
9.	place implant at convenient position w.r.t. arrows
10.	create vector figures

### Create Vector Field Figure
11.	determine a suitable camera angle, create a background image, plot the vector field with the same camera angle, adjust the size and location of the vector field w.r.t. the background


## Remarks
The code has two different (interlaced) parts: One part to analyze the field direction for the highest deposited power, and one part to analyze the field direction for the highest temperature rise per voxel. Three variables are used to specify which parts of the code will be executed. The value of “onlyExtract” determines whether the results are extracted (i.e. the worst directions are calculated), and “execute_visualizations” determines whether the visualization is set up. In principle, the whole code to analyze the deposited power can be executed by running only once, when both of these variables are true. In other words, the B field will always be extracted, but the simulation will only be (re)created when “onlyExtract” is not “True”. Additionally, the visualizations will be set up whenever “execute_visualizations” is “True”. Furthermore, when “execute_thermal” is “True”, the temperature analysis can be performed. This has to happen in two parts because the solver settings for the thermal simulations have to be changed manually (because it is currently not possible to set the solver processing unit to GPU via Python). This means that when “onlyExtract” is "False", the thermal simulations are set up, and when it is “True”, the results are extracted, but never both in the same run. The visualizations will only be set up when “onlyExtract” is “True” and “execute_visualizations” is also “True”.

