# sim4life_GC_WorstOrientation

The script allows to automatically compute the direction of a homogeneous, time-varying magnetic field, with unitary amplitude, that maximises the power deposited into an implant or the following maximum temperature increase. The analysis is in line with the test recommended by the ISO/TS 10974:2018 standard with regard to switched gradient field heating.<br>
The script requires Sim4Life together with a valid licence of the *Quasi-Static EM Solvers* and the *Thermal solver*.

## Operation

Once the script is run, it will prepare and execute three Magneto Quasi-Static simulations followed by six Thermal simulations. After the results have been computed by Sim4Life, the script analyses the simulation outcomes to print in the console the following data:

* Magnetic field direction (in polar coordinates with Theta: polar angle from z-axis, Phi azimuth angle from x-axis) that maximises the power deposition or temperature increase in the simulated model, namely the worst exposure condition;
* Components of the magnetic field vector leading to the worst exposure condition;
* Power deposited into the simulated model for the worst exposure condition;
* Maximum temperature increase obtained for the worst exposure condition after a specified exposure interval

In addition, the script creates a point in the model, placed where the maximum temperature increase occured

Finally, the following *numpy ndarrays* are returned:
* $M$ : *numpy ndarray* <br>
3 $\times$ 3 array. Given a magnetic field vector $B$ in Tesla, $M$ allows to compute the power deposited in the simulated model, $P$ as: <br>
$P = \frac{1}{2}B^TMB$
* $T$ :  *numpy ndarray* <br>
n_vox $\times$ 3 $\times$ 3 array with n_vox representing the number of voxels of the thermal simulation. Given a magnetic field vector $B$ in Tesla, $T$ allows to compute the temperature increase in *n*-th voxel of the simulated model, $\Delta T$ as: <br>
$\Delta T = B^TT[n-1]B$

## Usage

Before running the script, the model should be properly prepared following the listed steps:

1. Import the CAD of the implant that has to be tested;
2. Generate the bounding box of the implant. This can be done through the Sim4Life utility: Extract -> Bounding Box;
3. Create or import the phantom in which the implant has to be placed for the thermal experiments;
4. Generate the bounding box of the phantom. This can be done through the Sim4Life utility: Extract -> Bounding Box selecting the phantom entity;
5. Assign a material to each created or imported model entity. In the case some materials are not already included in the default material databases, a new database should be created contining the relevant materials;
6. Set the users parameters collected in the top part of the *computeWorstOrientation.py* script;
7. Run the *computeWorstOrientation.py* script.

## User Parameters

The following parameters should in the top part of the *computeWorstOrientation.py* script before running it:

* **onlyExtract** : *bool* <br>
If True the script doesn't execute any simulation only postprocessing the results of a previous script running;
* **model_embb_name** : *string* <br>
The name of the implant bounding box created in point 2;
* **excluded_from_em** : *list of string* <br>
A list containing the name of the entities that can be excluded from the electromagnetic (EM) computations (*e.g.* due to its low electrical conductivity, the phantom can be excluded from the EM computation);
* **temp_files_directory** : *string* <br>
Directory in which the auxialiary files are saved during the script execution;
* **material_database** : *list of string* <br>
A list containing the name of the material databases in which the script has to look for the material properties. This has to include also the databases created by the user in point 5;
* **em_voxel_size** : *numpy ndarray*
Three element array containing the size of the voxels (in mm) used to discretise the model during the EM simulations;
* **execute_thermal** : *bool*
If True the thermal analysis is performed after the EM one. Otherwise, the script will investigate only the direction of the magnetic field that maximise the deposited power;
* **model_thbb_name** : *string* <br>
The name of the phantom bounding box created in point 4;
* **th_voxel_size** : *numpy ndarray* <br>
Three element array containing the size of the voxels (in mm) used to discretise the model during the thermal simulations;
* **th_sim_interval** : *int* or *float* <br>
The time interval (in seconds) that is simulated in the thermal simulations. ISO/TS 10974:2018 suggests to perform 30 minutes experiments. Therefore this variable should be larger than 1800 s
* **th_sim_step_num** : *int* <br>
Number of steps stored during the thermal simulations from 0 s to *th_sim_interval*
* **th_snapshot** : *int* <br>
Reference snapshot (from 1 to `th_sim_step_num`) to perform the thermal assessments (*e.g.*, if it is equal to `th_sim_step_num`, the script will compute the magnetic field direction that maximise the temperature increase in the model after `th_sim_interval` seconds of exposure);
* **excluded_from_th_extr** : *list of string* <br>
A list containing the names of the entities that can be excluded by the thermal result analyses.

## Example

An example is made available in the examples folder of the repository. To run the example it is sufficient to open the *.smash* file with Sim4Lfe and run the *computeWorstOrientation.py* script from the Sim4Life scripter.