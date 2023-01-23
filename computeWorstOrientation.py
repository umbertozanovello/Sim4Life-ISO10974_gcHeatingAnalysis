# -*- coding: utf-8 -*-
import numpy as np
import h5py
import time
import s4l_v1.analysis as analysis
import s4l_v1.document as document
import s4l_v1.model as model
import s4l_v1.units as units
from s4l_v1 import ReleaseVersion
from s4l_v1 import Unit
import s4l_v1.simulation.emlf as emlf
import s4l_v1.simulation.thermal as thermal
import s4l_v1.materials.database as database


# USER PARAMETERS

onlyExtract = False # If True the script doesn't execute simulations
model_embb_name = "EM_BB_model"
excluded_from_em = ["Phantom"] # Entities to be excluded from em computations
temp_files_directory = "C:\\Simulazioni\\Zanovello\\Sim4Life\\STASIS\\WorstExposureComp\\temporary"
material_databases = ["User Default"]
em_voxel_size = np.array([1,1,1]) # mm

execute_thermal = True
model_thbb_name = "TH_BB_model"
phantom_name = "Phantom"
th_voxel_size = np.array([1,1,1]) # mm
th_sim_interval = 1800 # s
th_sim_step_num = 6 # Number of simulated step from 0 s to th_sim_interval
th_snapshot = 6 # Snapshot to be extracted to compute the thermal matrices (from 1)
excluded_from_th_extr = ["Phantom"] # The thermal matrices are not computed inside these entities

# Preliminary Settings

model.SetLengthUnits(units.MilliMeters)

#####################
## EM Simulations
#####################

def createEMSourceFiles():

	bb = model.AllEntities()[model_embb_name]
	bb_size = np.array([bb.Parameters[0].Value, bb.Parameters[1].Value, bb.Parameters[2].Value])
	bb_center = np.array(bb.Transform.Translation)
	
	with open(temp_files_directory+"\Bx.txt",'w') as fx, open(temp_files_directory+"\By.txt",'w') as fy, open(temp_files_directory+"\Bz.txt",'w') as fz:
		for i,f in enumerate([fx, fy, fz]):
			f.write("%.3f 2 %.3f\n" %((bb_center[0]-np.ceil(bb_size[0]/2))*1e-3, np.ceil(bb_size[0])*1e-3))
			f.write("%.3f 2 %.3f\n" %((bb_center[1]-np.ceil(bb_size[1]/2))*1e-3, np.ceil(bb_size[1])*1e-3))
			f.write("%.3f 2 %.3f\n\n" %((bb_center[2]-np.ceil(bb_size[2]/2))*1e-3, np.ceil(bb_size[2])*1e-3))
			
			for _ in range(7):
				f.write("0 0 0\n")
			else:
				f.write("0 0 0\n\n")
				
			for _ in range(8):
				f.write("%d %d %d\n" %tuple(np.eye(3)[i]))

def setEMSimulation(simName, vecPot_filename):

	# Define the version to use for default values
	ReleaseVersion.set_active(ReleaseVersion.version7_2)

	# Creating the simulation
	del document.AllSimulations[simName]
	simulation = emlf.MagnetoQuasiStaticSimulation()
	simulation.Name = simName
	
	# Editing QuasiStaticSetupSettings "Setup
	quasi_static_setup_settings = [x for x in simulation.AllSettings if isinstance(x, emlf.QuasiStaticSetupSettings) and x.Name == "Setup"][0]
	quasi_static_setup_settings.Frequency = 270.0, units.Hz
	
	sim_entities = []
	for entity in model.AllEntities():
		if (entity.Type == "ENTITY_TRIANGLEMESH" or entity.Type == "body") and (entity.Name not in [model_embb_name, model_thbb_name]) and (entity.Name not in excluded_from_em):
			sim_entities.append(entity)
	
	# Materials setting
	for entity in sim_entities:
		material_settings = simulation.AddMaterialSettings([entity])
		for material_database in material_databases:
			mat = database[material_database][entity.MaterialName]
			if mat is not None:
				break
		if mat is not None:
			simulation.LinkMaterialWithDatabase(material_settings, mat)
		else:
			print("%s not found in given databases" %entity.MaterialName)
	
	# Adding a new VectorPotentialSettings
	vector_potential_settings = simulation.AddVectorPotentialSettings([])
	vector_potential_settings.SourceType = vector_potential_settings.SourceType.enum.Userdef
	vector_potential_settings.UserDefFile = temp_files_directory + "//" + vecPot_filename
	vector_potential_settings.UseCubicInterpolation = False

	# Removing AutomaticGridSettings Automatic
	automatic_grid_settings = [x for x in simulation.AllSettings if isinstance(x, emlf.AutomaticGridSettings) and x.Name == "Automatic"][0]
	simulation.RemoveSettings(automatic_grid_settings)

	# Editing GlobalGridSettings
	global_grid_settings = simulation.GlobalGridSettings
	global_grid_settings.PaddingMode = global_grid_settings.PaddingMode.enum.Manual
	global_grid_settings.BottomPadding = np.array([0.0, 0.0, 0.0]), units.MilliMeters
	global_grid_settings.TopPadding = np.array([0.0, 0.0, 0.0]), units.MilliMeters

	# Adding a new ManualGridSettings
	manual_grid_settings = simulation.AddManualGridSettings([model.AllEntities()[model_embb_name]])
	manual_grid_settings.MaxStep = em_voxel_size, units.MilliMeters
	manual_grid_settings.Resolution = np.array([0.0, 0.0, 0.0]), units.MilliMeters

	# Editing AutomaticVoxelerSettings
	automatic_voxeler_settings = [x for x in simulation.AllSettings if isinstance(x, emlf.AutomaticVoxelerSettings) and x.Name == "Automatic Voxeler Settings"][0]
	simulation.Add(automatic_voxeler_settings, sim_entities)

	# Editing SolverSettings "Solver
	solver_settings = simulation.SolverSettings
	solver_settings.PredefinedTolerances = solver_settings.PredefinedTolerances.enum.High

	simulation.UpdateGrid()
	document.AllSimulations.Add( simulation )
	
	return simulation
	
def extractEMResults(simulation_name):

	simulation = document.AllSimulations[simulation_name]

	simulation_extractor = simulation.Results()

	em_sensor_extractor = simulation_extractor["Overall Field"]
	em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
	
	# Current density
	
	inputs = [em_sensor_extractor.Outputs["J(x,y,z,f0)"]]
	field_snapshot_filter = analysis.field.FieldSnapshotFilter(inputs=inputs)
	field_snapshot_filter.UpdateAttributes()
	field_snapshot_filter.Update()

	x = field_snapshot_filter.Outputs["J(x,y,z,f0)"].Data.Grid.XAxis
	y = field_snapshot_filter.Outputs["J(x,y,z,f0)"].Data.Grid.YAxis
	z = field_snapshot_filter.Outputs["J(x,y,z,f0)"].Data.Grid.ZAxis

	j = field_snapshot_filter.Outputs["J(x,y,z,f0)"].Data.Field(0)
	
	dX,dY,dZ = np.meshgrid(np.diff(x),np.diff(y),np.diff(z))

	vols = dX * dY * dZ
	vols = vols.flatten(order='F')
	
	# Electric field
	
	inputs = [em_sensor_extractor.Outputs["EM E(x,y,z,f0)"]]
	field_snapshot_filter = analysis.field.FieldSnapshotFilter(inputs=inputs)
	field_snapshot_filter.UpdateAttributes()
	field_snapshot_filter.Update()
	
	x = field_snapshot_filter.Outputs["EM E(x,y,z,f0)"].Data.Grid.XAxis
	y = field_snapshot_filter.Outputs["EM E(x,y,z,f0)"].Data.Grid.YAxis
	z = field_snapshot_filter.Outputs["EM E(x,y,z,f0)"].Data.Grid.ZAxis

	e = field_snapshot_filter.Outputs["EM E(x,y,z,f0)"].Data.Field(0)

	nan_mask = np.logical_not(np.isnan(np.sum(j, axis=1))) # True if not nan value
	
	return j[nan_mask], e[nan_mask], vols[nan_mask]

def computePowerWorstOrientation(M):
	
	worst_B = np.linalg.eigh(np.real(M))[1][:,2]
	max_power = 0.5 * np.linalg.eigh(np.real(M))[0][2]
	
	if worst_B[2] < 0:
		worst_B *= -1

	theta = np.arctan(np.sqrt(worst_B[0]**2+worst_B[1]**2)/worst_B[2]) # Polar angle (with respect to z-axis)
	phi = np.arctan2(worst_B[1],worst_B[0])    

	print("POWER WORST DIRECTION:\nTheta: %.2f°, Phi: %.2f°, worst B: %.2f, %.2f, %.2f, Max power: %.2f W" %(np.rad2deg(theta), np.rad2deg(phi), worst_B[0], worst_B[1], worst_B[2], max_power))
		
	return worst_B

#######################
## Thermal Simulations
#######################

def createThSourceFiles(jx, ex, jy, ey, jz, ez):
	
	# Exporting of data cache from EM simulations
	simulation = document.AllSimulations["Bx"]
	simulation_extractor = simulation.Results()
	
	em_sensor_extractor = simulation_extractor["Overall Field"]
	em_sensor_extractor.FrequencySettings.ExtractedFrequency = u"All"
	
	inputs = [em_sensor_extractor.Outputs["El. Loss Density(x,y,z,f0)"]]
	field_snapshot_filter = analysis.field.FieldSnapshotFilter(inputs=inputs)
	field_snapshot_filter.UpdateAttributes()
	
	inputs = [field_snapshot_filter.Outputs["El. Loss Density(x,y,z,f0)"]]
	data_cache_exporter = analysis.exporters.DataCacheExporter(inputs=inputs)
	data_cache_exporter.FileName = temp_files_directory + "//heatSourceCache_orig.cache"
	data_cache_exporter.UpdateAttributes()
	data_cache_exporter.Update(overwrite=True)
	
	# Preparing all cache files: creating files
	with open(temp_files_directory + "//heatSourceCache_orig.cache", "rb") as f_source:
		for f_name in ["xx", "yy", "zz", "xy", "xz", "yz"]:
			with open(temp_files_directory + "//heatSourceCache_%s.cache" %f_name, "wb") as f_dest:
				f_source.seek(0)
				f_dest.write(f_source.read())
	
	# Preparing all cache files: writing power density distributions
	
	# xx
	with h5py.File(temp_files_directory + "//heatSourceCache_xx.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ex.conj(), jx) #pd_j = ex.conj()_ji * jx_ji with sum only on i index
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)
	# yy
	with h5py.File(temp_files_directory + "//heatSourceCache_yy.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ey.conj(), jy)
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)
	# zz
	with h5py.File(temp_files_directory + "//heatSourceCache_zz.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ez.conj(), jz)
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)
	# xy
	with h5py.File(temp_files_directory + "//heatSourceCache_xy.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ex.conj(), jy)
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)	
	# xz
	with h5py.File(temp_files_directory + "//heatSourceCache_xz.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ex.conj(), jz)
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)
	# yz
	with h5py.File(temp_files_directory + "//heatSourceCache_yz.cache","r+") as ifile:
		pd = 0.5*np.einsum('ji,ji->j',ey.conj(), jz)
		r = ifile["datacache"]["_Object"]["0"]["_Object"]["_Field_0"]
		r2 = r[:,0]
		r2 = r2.astype(complex)
		r2[np.logical_not(np.isnan(r2))] = pd
		r2[np.isnan(r2)] = 0
		r[:,0] = np.real(r2)
		
def setThermalSimulation(simName, cache_filename):
	# Creating the simulation
	del document.AllSimulations[simName]
	simulation = thermal.TransientSimulation()
	simulation.Name = simName

	sim_entities = []
	for entity in model.AllEntities():
		if (entity.Type == "ENTITY_TRIANGLEMESH" or entity.Type == "body") and (entity.Name not in [model_embb_name, model_thbb_name]):
			sim_entities.append(entity)
			
	# Editing TransientSetupSettings "Setup
	transient_setup_settings = [x for x in simulation.AllSettings if isinstance(x, thermal.TransientSetupSettings) and x.Name == "Setup"][0]
	transient_setup_settings.SimulationTime = th_sim_interval, units.Seconds

	# Materials setting
	for entity in sim_entities:
		material_settings = simulation.AddMaterialSettings([entity])
		for material_database in material_databases:
			mat = database[material_database][entity.MaterialName]
			if mat is not None:
				break
		if mat is not None:
			simulation.LinkMaterialWithDatabase(material_settings, mat)
		else:
			print("%s not found in given databases" %entity.MaterialName)
			
	# Editing GlobalInitialConditionSettings "Initial Conditions
	global_initial_condition_settings = simulation.GlobalInitialConditionSettings
	global_initial_condition_settings.OverallTemperature = 0.0, units.Coulombs

	# Adding a new TransientHeatSourceSettings
	transient_heat_source_settings = simulation.AddHeatSourceSettings([])
	transient_heat_source_settings.SourceType = transient_heat_source_settings.SourceType.enum.DataObject
	transient_heat_source_settings.SourceDataObject.CacheFile = temp_files_directory + "//" + cache_filename

	# Editing BoundaryConditionSettings "Boundary Settings
	boundary_condition_settings = [x for x in simulation.AllSettings if isinstance(x, thermal.BoundaryConditionSettings) and x.Name == "Boundary Settings"][0]
	components = [simulation.AllComponents["Background"]]
	simulation.Add(boundary_condition_settings, components)
	boundary_condition_settings.BoundaryType = boundary_condition_settings.BoundaryType.enum.Neumann
	
	# Editing TransientFieldSensorSettings "Sensor Settings
	transient_field_sensor_settings = [x for x in simulation.AllSettings if isinstance(x, thermal.TransientFieldSensorSettings) and x.Name == "Sensor Settings"][0]
	components = [simulation.AllComponents["Overall Field"]]
	simulation.Add(transient_field_sensor_settings, components)
	transient_field_sensor_settings.MaximumNoSnapshots = th_sim_step_num

	# Adding a new ManualGridSettings
	manual_grid_settings = simulation.AddManualGridSettings([model.AllEntities()[model_thbb_name]])
	manual_grid_settings.MaxStep = th_voxel_size, units.MilliMeters

	# Editing AutomaticVoxelerSettings "Automatic Voxeler Settings
	automatic_voxeler_settings = [x for x in simulation.AllSettings if isinstance(x, thermal.AutomaticVoxelerSettings) and x.Name == "Automatic Voxeler Settings"][0]
	components = [x for x in sim_entities if x.Name != phantom_name]
	automatic_voxeler_settings.Priority = 1
	simulation.Add(automatic_voxeler_settings, components)

	automatic_voxeler_settings = thermal.AutomaticVoxelerSettings()
	components = [model.AllEntities()[phantom_name]]
	automatic_voxeler_settings.Name = "Automatic Voxeler Settings Phantom"
	automatic_voxeler_settings.Priority = 0
	simulation.Add(automatic_voxeler_settings, components)

	# Update the materials with the new frequency parameters
	simulation.UpdateAllMaterials()

	# Update the grid with the new parameters
	simulation.UpdateGrid()

	# Add the simulation to the UI
	document.AllSimulations.Add( simulation )
	
	return simulation

def extractThermalResults(simulation_name):

	simulation = document.AllSimulations[simulation_name]

	simulation_extractor = simulation.Results()

	th_sensor_extractor = simulation_extractor["Overall Field"]

	extr_entities = []

	for entity in model.AllEntities():
		if (entity.Type == "ENTITY_TRIANGLEMESH" or entity.Type == "body") and (entity.Name not in [model_embb_name, model_thbb_name]) and (entity.Name not in excluded_from_th_extr):
			extr_entities.append(entity)

	inputs = [th_sensor_extractor.Outputs["T(x,y,z,t)"]]
	field_masking_filter = analysis.core.FieldMaskingFilter(inputs=inputs)
	field_masking_filter.UseNaN = True
	field_masking_filter.SetAllMaterials(False)
	for entity in extr_entities:
		field_masking_filter.SetEntities([entity])
	field_masking_filter.UpdateAttributes()
	field_masking_filter.Update()
	field_masking_filter.Update()
	
	temp = field_masking_filter.Outputs["T(x,y,z,t)"].Data.Field(th_snapshot-1)[:,0]
	
	x = field_masking_filter.Outputs["T(x,y,z,t)"].Data.Grid.XAxis
	x = 0.5*(x[1:]+x[:-1])
	y = field_masking_filter.Outputs["T(x,y,z,t)"].Data.Grid.YAxis
	y = 0.5*(y[1:]+y[:-1])
	z = field_masking_filter.Outputs["T(x,y,z,t)"].Data.Grid.ZAxis
	z = 0.5*(z[1:]+z[:-1])

	X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
	coords = np.array([X.flatten(order='F'),Y.flatten(order='F'),Z.flatten(order='F')]).T # Shape: nVoxel x 3
	nan_mask = np.logical_not(np.isnan(temp))
	
	return temp[nan_mask], coords[nan_mask,:]

def computeTemperatureWorstOrientation(T, coords):
	
	worst_B = None
	maxmax_temp = 0
	max_T_coords = None # meters
	
	for i,voxel_T in enumerate(T):
		eigval, eigvect = np.linalg.eigh(voxel_T)
		if eigval[2] > maxmax_temp:
			maxmax_temp = eigval[2]
			worst_B = eigvect[:,2]
			max_T_coords = coords[i]
	
	if worst_B[2] < 0:
		worst_B *= -1
		
	theta = np.arctan(np.sqrt(worst_B[0]**2+worst_B[1]**2)/worst_B[2]) # Polar angle (with respect to z-axis)
	phi = np.arctan2(worst_B[1],worst_B[0])    

	print("TEMPERATURE WORST DIRECTION:\nTheta: %.2f°, Phi: %.2f°, worst B: %.2f, %.2f, %.2f, Worst temperature: %.2f °C" %(np.rad2deg(theta), np.rad2deg(phi), worst_B[0], worst_B[1], worst_B[2], maxmax_temp))
	
	bb = model.AllEntities()[model_thbb_name]
	bb_size = np.array([bb.Parameters[0].Value, bb.Parameters[1].Value, bb.Parameters[2].Value])
	bb_center = np.array(bb.Transform.Translation)
	
	point_coords = max_T_coords*1e3
	max_temp_point = model.CreatePoint(model.Vec3(list(point_coords)))
	max_temp_point.Name = "Worst_temperature_point"
		
	return worst_B
	
def main():
	#EM analysis
	
	if not onlyExtract:
		# Input field preparation
		createEMSourceFiles()
		
		# Simulations preparation and execution
		for orientation in ["Bx", "By", "Bz"]:
			simulation = setEMSimulation(orientation, orientation+".txt")
			simulation.CreateVoxels()
			simulation.RunSimulation()
			while not simulation.HasResults():
				time.sleep(1)
	
	# Results analysis
	jx,ex,_ = extractEMResults("Bx")
	jy,ey,_ = extractEMResults("By")
	jz,ez,vols = extractEMResults("Bz")
	
	J = np.concatenate((jx[:,:,None],jy[:,:,None],jz[:,:,None]),axis=2)
	E = np.concatenate((ex[:,:,None],ey[:,:,None],ez[:,:,None]),axis=2)

	M = np.sum((np.transpose(E.conj(),axes=(0,2,1)) @ J).T * vols, axis=2)
		
	#Thermal analysis
	
	if execute_thermal:
		if not onlyExtract:
			# Input powerDensity preparation
			createThSourceFiles(jx,ex,jy,ey,jz,ez)
			
			# Set and execution of simulations
			for element in ["xx", "yy", "zz", "xy", "xz", "yz"]:
				simulation = setThermalSimulation(element, "heatSourceCache_"+element+".cache")
				simulation.CreateVoxels()
				simulation.RunSimulation()
				while not simulation.HasResults():
					time.sleep(10)
				
		# Results extraction
		txx,_ = extractThermalResults("xx")
		tyy,_ = extractThermalResults("yy")
		tzz,_ = extractThermalResults("zz")
		txy,_ = extractThermalResults("xy")
		txz,_ = extractThermalResults("xz")
		tyz,coords = extractThermalResults("yz")
	
		T = np.array([[txx, txy, txz],[txy, tyy, tyz],[txz, tyz, tzz]]).T

	worst_B_power = computePowerWorstOrientation(M)
	
	if execute_thermal:
		worst_B_temp = computeTemperatureWorstOrientation(T, coords)
		return M, T
	
	return M
	
		
if __name__ == "__main__":
	M, T = main()
