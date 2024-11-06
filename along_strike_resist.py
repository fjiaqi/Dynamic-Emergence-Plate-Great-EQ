import logging
import underworld as uw
import underworld.function as fn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import csv
import sys
sys.path.append('./model/')
sys.path.append('./utils/')
from references import *
from domain import *
from materials import *
from structures import *
from mesh_utils import *
from mpi_utils import *
from diag_utils import *

plt.rcParams['font.size'] = 16

real_resist_length = float(sys.argv[1])
real_phi_tr = float(sys.argv[2])
restart_step = 116
output_dir = Path('/home1/08524/jqfang/scratch/model_2411/2.5d/')
basicinput_dir = Path('/home1/08524/jqfang/scratch/model_2411/2.5d_ic/basic/')
input_dir = Path('/home1/08524/jqfang/scratch/model_2411/2.5d_ic/velocity/')
restart_flag = 's'
initial_guess = False
initial_guess_dir = Path('./output/')

# Create result directories and log
basic_dir = output_dir / 'basic'
velocity_dir = output_dir / 'velocity'
slipv_stress_dir = output_dir / 'slipv_stress'
figure_dir = output_dir / 'figures'
if(uw.mpi.rank == 0):
    output_dir.mkdir(exist_ok=True, parents=True)
    basic_dir.mkdir(exist_ok=True, parents=True)
    velocity_dir.mkdir(exist_ok=True, parents=True)
    slipv_stress_dir.mkdir(exist_ok=True, parents=True)
    figure_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("progress_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(output_dir / f"progress_L{real_resist_length:.0e}.log", mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(f'Started using {uw.mpi.size} threads')


xres = 1024
zres = 256
yres = 2

# Mesh
if(restart_step < -1):
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, yres, zres), 
                                        minCoord    = (-domain_width_west, 0, -domain_depth), 
                                        maxCoord    = ( domain_width_east, domain_length,  air_depth) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, zres), 
                                        minCoord    = (-domain_width_west, -domain_depth), 
                                        maxCoord    = ( domain_width_east,  air_depth) )

    refine_left_x = -250e3 / reference_length
    refine_right_x = 250e3 / reference_length
    refine_buffer_x = 0e3 / reference_length
    refine_ratio_x = 4.

    refine_left_z = np.maximum(west_lithos_bottom_coords.min(), east_lithos_bottom_coords.min())
    refine_right_z = air_depth
    refine_buffer_z = 0e3 / reference_length
    refine_ratio_z = 3.

    mesh_deform_data = mesh.data.copy()
    mesh_deform_data[:, 0] = deform_map(mesh.data[:, 0], -domain_width_west, domain_width_east, 
                                        refine_left_x, refine_right_x, refine_buffer_x, refine_ratio_x)
    mesh_deform_data[:, -1] = deform_map(mesh.data[:, -1], -domain_depth, air_depth, 
                                        refine_left_z, refine_right_z, refine_buffer_z, refine_ratio_z)
    with mesh.deform_mesh(isRegular=False):
        mesh.data[:] = mesh_deform_data
else:
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, yres, zres),
                                         minCoord    = (0., 0., 0.),
                                         maxCoord    = (1., 1., 1.) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, zres) )
    mesh.load(str(basicinput_dir / 'mesh_0.h5'))

# Velocity, pressure, temperature
velocityField    = mesh.add_variable(         nodeDofCount=mesh.dim )
pressureField    = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField = mesh.add_variable(         nodeDofCount=1 )

# Swarm
swarm = uw.swarm.Swarm( mesh=mesh )
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
swarm.allow_parallel_nn = True

# Setup temperature and material structure
if(restart_step < -1):
    velocityField.data[:] = 0.
    pressureField.data[:] = 0.
    temperatureField.data[:] = 0. 

    bend_buffer_width = 80e3 / reference_length
    slab_front_buffer_width = 80e3 / reference_length

    slab_top_x = slab_top_zxfunc(mesh.data[:, -1])
    slab_top_z = slab_top_xzfunc(mesh.data[:, 0])
    slab_front_z = slab_front_xzfunc(mesh.data[:, 0])
    for idx in range(mesh.data.shape[0]):
        xm = mesh.data[idx, 0]
        zm = mesh.data[idx, -1]
        if(zm > 0.):
            temperatureField.data[idx] = 0.
        elif(xm < ridge_loc):
            temperatureField.data[idx] = 1.
        elif(xm >= ridge_loc and xm < 0.):
            tx = np.minimum((xm-ridge_loc)/vel_west_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm >= 0
        and xm <= slab_front_coords[:, 0].max()*1.5
        and zm <= 0):
            xs = slab_top_x[idx]
            zs = slab_top_z[idx]
            zf = slab_front_z[idx]
            if(xm <= xs
            and zm <= zs
            and zm >= zf):
                pt_m = shg.Point(xm, zm)
                xd = -slab_top_line.distance(pt_m)
                tp = slab_top_line.project(pt_m)/vel_west_plate
                if(xm < bend_buffer_width):
                    buffer_coef = xm/bend_buffer_width
                    xd = (zm-zs)*(1.-buffer_coef) + xd*buffer_coef
                    pt_tmp = shg.Point(xm, zs)
                    tp = slab_top_line.project(pt_tmp)/vel_west_plate*(1-buffer_coef) + tp*buffer_coef
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm >= slab_top_coords[:, 1].min()):
                xd = xm-xs
                pt_tmp = shg.Point(xs, zm)
                tp = slab_top_line.project(pt_tmp)/vel_west_plate
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm <= xs
                and zm <= zs
                and zm < zf):
                xd = (xm-xs)*np.cos(slab_front_slope_rad)
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((zf-zm)*np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm < slab_top_coords[:, 1].min()):
                xd = xm-xs
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((slab_top_coords[:, 1].min()-zm)/np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            else:
                tx = np.minimum(age_east_plate, age_max_cut)
                temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm > slab_front_coords[:, 0].max()*1.2):
            tx = np.minimum(age_east_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        else:
            temperatureField.data[idx] = 1.

    temperatureField.data[temperatureField.data > 0.99] = 1.
    temperatureField.data[:] = np.clip(temperatureField.data, 0., 1.)

    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=16 )
    swarm.populate_using_layout( layout=swarmLayout )

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust]

    west_lithos_bottom_z = west_lithos_bottom_xzfunc(swarm.data[:, 0])
    shear_zone_bottom_z = shear_zone_bottom_xzfunc(swarm.data[:, 0])
    slab_top_x = slab_top_zxfunc(swarm.data[:, -1])
    slab_top_z = slab_top_xzfunc(swarm.data[:, 0])
    slab_front_z = slab_front_xzfunc(swarm.data[:, 0])
    east_lithos_bottom_z = east_lithos_bottom_xzfunc(swarm.data[:, 0])
    for idx in range(swarm.data.shape[0]):
        xp = swarm.data[idx, 0]
        zp = swarm.data[idx, -1]
        if(zp > 0.):
            materialIndex.data[idx] = air.index
        elif(zp >= shear_zone_bottom_z[idx]
            and zp <= slab_top_z[idx]
            and zp >= slab_front_z[idx]
            and zp >= east_lithos_bottom_coords.min()):
            materialIndex.data[idx] = shear_zone.index
        elif(xp >= ridge_loc
            and zp > west_lithos_bottom_z[idx]*0.5
            and zp > -west_crust_depth
            and zp < shear_zone_bottom_z[idx]):
            materialIndex.data[idx] = west_crust.index
        elif(xp >= ridge_loc
            and zp >= west_lithos_bottom_z[idx]
            and zp >= slab_front_z[idx]
            and zp < shear_zone_bottom_z[idx]):
            materialIndex.data[idx] = west_plate.index
        elif(xp >= slab_top_x[idx]
            and zp > -east_crust_depth):
            materialIndex.data[idx] = east_crust.index
        elif(xp >= slab_top_x[idx] 
            and zp >= east_lithos_bottom_z[idx]):
            materialIndex.data[idx] = east_plate.index
        else:
            materialIndex.data[idx] = mantle.index

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.data[:] = 0.
else:
    velocityField.load(str(input_dir / f'velocity_{restart_flag}{restart_step}.h5'))
    pressureField.load(str(input_dir / f'pressure_{restart_flag}{restart_step}.h5'))
    temperatureField.load(str(basicinput_dir / f'temperature_0.h5'))

    swarm.load(str(basicinput_dir / f'swarm_0.h5'))

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust]

    materialIndex.load(str(basicinput_dir / f'material_0.h5'))

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.load(str(input_dir / f'prestress_{restart_flag}{restart_step}.h5'))

# Mesh stats
dx_local = np.diff(np.unique(mesh.data[:, 0]))
dz_local = np.diff(np.unique(mesh.data[:, -1]))
if(yres > 0):
    dy_local = np.diff(np.unique(mesh.data[:, 1]))
else:
    dy_local = np.array([0.])
dxdydz_local_min = np.array([dx_local.min(), dy_local.min(), dz_local.min()])
dxdydz_local_max = np.array([dx_local.max(), dy_local.max(), dz_local.max()])
dxdydz_min = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_min, dxdydz_min, op=MPI.MIN)
dxdydz_max = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_max, dxdydz_max, op=MPI.MAX)

if(uw.mpi.rank == 0):
    msg = f'xres: {xres}, dx: {l_km(dxdydz_min[0]):.3e} - {l_km(dxdydz_max[0]):.3e} km\n'
    msg += f'yres: {yres}, dy: {l_km(dxdydz_min[1]):.3e} - {l_km(dxdydz_max[1]):.3e} km\n'
    msg += f'zres: {zres}, dz: {l_km(dxdydz_min[2]):.3e} - {l_km(dxdydz_max[2]):.3e} km'
    logger.debug(msg)

step = 0
time = 0.

# Boundary conditions
appliedTraction = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=mesh.dim )
appliedTraction.data[:] = 0.

if(yres > 0):
    topWall = mesh.specialSets["MaxK_VertexSet"]
    bottomWall = mesh.specialSets["MinK_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    frontWall = mesh.specialSets["MinJ_VertexSet"]
    backWall = mesh.specialSets["MaxJ_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = frontWall + backWall
    kWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls, kWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls+kWalls, iWalls+kWalls, iWalls+jWalls) )
else:
    topWall = mesh.specialSets["MaxJ_VertexSet"]
    bottomWall = mesh.specialSets["MinJ_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls, iWalls) )

# Surface swarm for tracking
surfaceSwarm = uw.swarm.Swarm(mesh)
advector_surface = uw.systems.SwarmAdvector( swarm=surfaceSwarm, velocityField=velocityField, order=2 )
xres_surface = 501
if(yres > 0):
    yres_surface = yres+1
    surfacePoints = np.zeros((xres_surface*yres_surface, mesh.dim))
    x_surface = np.linspace(-domain_width_west, domain_width_east, xres_surface)
    y_surface = np.linspace(0, domain_length, yres_surface)
    xx_surface, yy_surface = np.meshgrid(x_surface, y_surface)
    surfacePoints[:, 0] = xx_surface.ravel()
    surfacePoints[:, 1] = yy_surface.ravel()
else:
    surfacePoints = np.zeros((xres_surface, mesh.dim))
    surfacePoints[:, 0] = np.linspace(-domain_width_west, domain_width_east, xres_surface)
if(air_depth > 0.):
    surfacePoints[:, -1] = -1e3 / reference_length
else:
    surfacePoints[:, -1] = 0.
surface_particles = surfaceSwarm.add_particles_with_coordinates( surfacePoints )

# Buoyancy
boussinesq = 1.

densityMap = { material.index : material.density - boussinesq for material in material_list }
densityFn  = fn.branching.map( fn_key=materialIndex, mapping=densityMap )

diffusivityMap = { material.index : material.diffusivity for material in material_list }
diffusivityFn = fn.branching.map( fn_key=materialIndex, mapping=diffusivityMap )

alphaMap = { material.index : material.alpha for material in material_list }
alphaFn = fn.branching.map( fn_key=materialIndex, mapping=alphaMap )

unit_z = np.zeros(mesh.dim)
unit_z[-1] = -1.
buoyancyFn = (Rb * densityFn - Ra * temperatureField * alphaFn) * unit_z

# Viscosity, elasticity, friction
maxViscosityMap = { material.index : material.max_viscosity for material in material_list }
maxViscosity = fn.branching.map( fn_key=materialIndex, mapping=maxViscosityMap )

minViscosityMap = { material.index : material.min_viscosity for material in material_list }
minViscosity = fn.branching.map( fn_key=materialIndex, mapping=minViscosityMap )

diffEMap = { material.index : material.diff_E for material in material_list }
diffE = fn.branching.map( fn_key=materialIndex, mapping=diffEMap )

diffVMap = { material.index : material.diff_V for material in material_list }
diffV = fn.branching.map( fn_key=materialIndex, mapping=diffVMap )

diffAMap = { material.index : material.diff_A for material in material_list }
diffA = fn.branching.map( fn_key=materialIndex, mapping=diffAMap )

dislNMap = { material.index : material.disl_n for material in material_list }
dislN = fn.branching.map( fn_key=materialIndex, mapping=dislNMap )

dislEMap = { material.index : material.disl_E for material in material_list }
dislE = fn.branching.map( fn_key=materialIndex, mapping=dislEMap )

dislVMap = { material.index : material.disl_V for material in material_list }
dislV = fn.branching.map( fn_key=materialIndex, mapping=dislVMap )

dislAMap = { material.index : material.disl_A for material in material_list }
dislA = fn.branching.map( fn_key=materialIndex, mapping=dislAMap )

shearModulusHighMap = { material.index : material.shear_modulus_high for material in material_list }
shearModulusHigh = fn.branching.map( fn_key=materialIndex, mapping=shearModulusHighMap )

shearModulusLowMap = { material.index : material.shear_modulus_low for material in material_list }
shearModulusLow = fn.branching.map( fn_key=materialIndex, mapping=shearModulusLowMap )

cohesionMap = { material.index : material.cohesion for material in material_list }
cohesionFn = fn.branching.map( fn_key=materialIndex, mapping=cohesionMap )

maxYieldStressMap = { material.index : material.max_yield_stress for material in material_list }
maxYieldStress = fn.branching.map( fn_key=materialIndex, mapping=maxYieldStressMap )

strainRate = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2nd_Invariant = fn.tensor.second_invariant(strainRate)
tiny_strain_rate = 1e-25 * reference_time
slipVelocity = 2. * strainRate_2nd_Invariant * shear_zone_width

depth = fn.misc.max(-fn.input()[mesh.dim-1], 0.)
lithostaticPressure = fn.misc.max(Rb*depth, 0.)
trueLithostaticPressure = reference_density * gravity * depth*reference_length
trueTemperature = reference_T0 + temperatureField * (reference_T1-reference_T0)

friction_st = 0.6
friction_dyn = np.array([0.8, 0.2, 0.8])
friction_st_lithos = 0.6
friction_dyn_lithos = 0.6
friction_trans_true_temp = np.array([373., 423., 623., 723.])
frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                            / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                            * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                            / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                            * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
frictionSt = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_st_lithos),
                                         (materialIndex > shear_zone.index, friction_st_lithos),
                                         (True, friction_st) ] )
frictionDyn = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                          (materialIndex > shear_zone.index, friction_dyn_lithos),
                                          (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                          (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                          (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                          (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                          (True, friction_dyn[2]) ] )
ref_slip_velocity = 10e-2 / s_per_year / reference_velocity
refSlipVelocity = fn.misc.constant(ref_slip_velocity)
frictionFn = frictionDyn + (frictionSt-frictionDyn)/(1.+slipVelocity/refSlipVelocity)

factor_lithos = 0.
factor_shear_zone = 0.95
factor_trans_depth = np.array([60e3, 80e3]) / reference_length
factorTransition = ((depth-factor_trans_depth[0])
                     / (factor_trans_depth[1]-factor_trans_depth[0])
                     * (factor_lithos-factor_shear_zone)) + factor_shear_zone
porePressureFactor = fn.branching.conditional( [ (materialIndex < shear_zone.index, factor_lithos),
                                                 (materialIndex > shear_zone.index, factor_lithos),
                                                 (depth <= factor_trans_depth[0], factor_shear_zone),
                                                 (depth <= factor_trans_depth[1], factorTransition),
                                                 (True, factor_lithos) ] )

normalStress = (1. - porePressureFactor) * lithostaticPressure

truePressure = (1. - porePressureFactor) * trueLithostaticPressure
diffViscosity = diffA * fn.math.exp((diffE+truePressure*diffV) / (R_const*trueTemperature))
dislViscosity = dislA * (fn.math.exp((dislE+truePressure*dislV) / (dislN*R_const*trueTemperature)) 
                         * fn.math.pow(strainRate_2nd_Invariant+tiny_strain_rate, 1./dislN-1.))
pureViscosity = 1./(1./diffViscosity + 1./dislViscosity)
viscosityFn = fn.misc.max(fn.misc.min(pureViscosity, maxViscosity), minViscosity)

shearModulusFn = shearModulusHigh + (shearModulusLow - shearModulusHigh) * fn.math.pow(temperatureField, 2.)
maxwellTime = viscosityFn / shearModulusFn

# Function for diagnose at each step
def diagnose(**kwargs):
    surf_vars = getGlobalSwarmVar(np.hstack((surfaceSwarm.data,
                                             velocityField.evaluate(surfaceSwarm.data))))
    surf_loc = surf_vars[:, 0:mesh.dim]
    surf_vel = surf_vars[:, mesh.dim:2*mesh.dim]

    pt_loc, pt_vars = getGlobalMeshVar(mesh, np.hstack((materialIndex.evaluate(mesh),
                                                        slipVelocity.evaluate(mesh))))
    
    pt_material = pt_vars[:, 0]
    pt_slip_vel = pt_vars[:, 1]

    if(uw.mpi.rank == 0):
        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]
        plate_vel_quantity = plate_quantity(surf_vel[:, 0], surf_loc[:, 0],
                                            ridge_loc, 0.,
                                            return_std=True,
                                            return_max=True,
                                            return_rms=True)

        slip_vel_quantity = shear_zone_quantity(pt_slip_vel, pt_material,
                                                return_std=True,
                                                return_max=True,
                                                return_rms=True)

        msg = f'Step, time (yr)\n'
        msg += f'{step:d} {t_yr(time):.3f}\n'
        msg += f'Plate velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(plate_vel_quantity)):
            msg += f'{v_mm_yr(plate_vel_quantity[i]):.3e} '
        msg += '\n'
        msg += f'Slip velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(slip_vel_quantity)):
            msg += f'{v_mm_yr(slip_vel_quantity[i]):.3e} '
        logger.debug(msg)

        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]

        if(kwargs.get('plot') is not None):
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]))
            ax.scatter(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]), s=1)
            ax.set_title(f'Step = {step}')
            ax.set_xlim(-l_km(domain_width_west), l_km(domain_width_east))
            ax.set_xlabel('x (km)')
            ax.set_ylabel('vx (mm/yr)')
            ax.grid()
            ax.minorticks_on()
            plt.tight_layout()
            plt.savefig(figure_dir / kwargs.get('plot'))
            plt.close()

        diagnostics = np.r_[plate_vel_quantity, slip_vel_quantity]
    else:
        diagnostics = np.zeros(8)

    MPI.COMM_WORLD.Bcast(diagnostics, root=0)
    return diagnostics

# Solve for long-term (creating prestress IC for short-term)
if(restart_step < -1):
    flag = 'l'

    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    dt_e_ratio = 0.1
    dt_e = west_plate.max_viscosity/west_plate.shear_modulus_high * dt_e_ratio

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress
    stressFn_2nd_Invariant = fn.tensor.second_invariant(stressFn)

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    nsteps_long = 51
    step = 0
    time = 0.

    solver.set_inner_method("mumps")
    solver.set_penalty(1e3)

    while(step < nsteps_long):
        solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )

        diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        
        dt = dt_e*phi_dt

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        step = step + 1
        time = time + dt

# Solve short-term
flag = 's'

longTermVelocity = velocityField.copy()
longTermVelocity.data[:] = velocityField.data[:]

previousVelocity = velocityField.copy()
temporaryVelocity = velocityField.copy()
previousTraction = appliedTraction.copy()
temporaryTraction = appliedTraction.copy()
expectedTraction = appliedTraction.copy()

volume_integral = uw.utils.Integral(1., mesh)
volume = volume_integral.evaluate()[0]
front_area_integral = uw.utils.Integral(1., mesh, 'surface', frontWall)
front_area = front_area_integral.evaluate()[0]

yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

dt_e = 5. * s_per_year / reference_time

strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                            (dislViscosity < diffViscosity, 1), 
                                            (True, 2) ] )
viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

viscousStress      = 2. * viscosityEff * strainRate
tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
viscoelasticStress = 2. * viscosityEff * strainRateEff
stressFn           = viscoelasticStress
stressFn_2nd_Invariant = fn.tensor.second_invariant(stressFn)

stokes = uw.systems.Stokes( velocityField = velocityField,
                            pressureField = pressureField,
                            voronoi_swarm = swarm,
                            conditions    = [velBC, trBC],
                            fn_viscosity  = viscosityEff,
                            fn_bodyforce  = buoyancyFn,
                            fn_stresshistory = tauHistoryFn )

solver = uw.systems.Solver( stokes )


phi_dt = 1.
dt = dt_e*phi_dt

nsteps_short = 6
step = 0
time = 0.

solver.set_inner_method("mumps")
solver.set_penalty(1e3)

resist_length = real_resist_length / reference_length
phi_tr = real_phi_tr
event_vel = vel_west_plate * 1.5
nits = 1000
residual_tol = 1e-2
nits_min = 1

resistShearModulus = viscosityEff / dt

solver_flag = 100
maskinit_nsteps = -1
postevent_nsteps = 5
preevent_tol = 1e-2
coevent_tol = 1e-3
postevent_tol = 2e-4

while(step < nsteps_short):
    if(solver_flag >= postevent_nsteps):
        solver.solve( nonLinearIterate=True, nonLinearTolerance=preevent_tol, nonLinearMaxIterations=500 )
    else:
        solver.solve( nonLinearIterate=True, nonLinearTolerance=postevent_tol, nonLinearMaxIterations=500 )

    diagnostics = diagnose(plot=f'vx_{flag}{step}.png')

    if(diagnostics[2] > event_vel and solver_flag >= postevent_nsteps and step > maskinit_nsteps):
        solver_flag = 0
        if(uw.mpi.rank == 0):
            logger.debug(f'Event detected, continue solving!!!!!!!!!!')
        solver.solve( nonLinearIterate=True, nonLinearTolerance=coevent_tol, nonLinearMaxIterations=500 )
        diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
    else:
        solver_flag += 1

    # Apply traction on the front wall if an event is detected
    if(diagnostics[2] > event_vel):
        if(uw.mpi.rank == 0):
            logger.debug(f'Event detected, traction applied!!!!!!!!!!')
        previousVelocity.data[:] = velocityField.data[:]
        previousTraction.data[:] = appliedTraction.data[:]

        if(initial_guess):
            velocityField.load(str(initial_guess_dir / f'velocity_s{step}.h5'))
            appliedTraction.load(str(initial_guess_dir / f'applied_traction_s{step}.h5'))

        expectedTraction.data[frontWall, 0] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                ((longTermVelocity.data[frontWall, 0] 
                                                - velocityField.data[frontWall, 0]) / resist_length
                                                + previousTraction.data[frontWall, 0]
                                                / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
        expectedTraction.data[frontWall, -1] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                ((longTermVelocity.data[frontWall, -1] 
                                                - velocityField.data[frontWall, -1]) / resist_length
                                                + previousTraction.data[frontWall, -1]
                                                / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
        
        ii = 0
        while(ii < nits):
            if(uw.mpi.rank == 0):
                msg = f'Iteration, phi\n'
                msg += f'{ii+1:d} {phi_tr:.6f}'
                logger.debug(msg)
            temporaryVelocity.data[:] = velocityField.data[:]
            temporaryTraction.data[:] = appliedTraction.data[:]

            appliedTraction.data[frontWall, 0] = phi_tr * expectedTraction.data[frontWall, 0] + \
                                                        (1.-phi_tr) * appliedTraction.data[frontWall, 0]
            appliedTraction.data[frontWall, -1] = phi_tr * expectedTraction.data[frontWall, -1] + \
                                                        (1.-phi_tr) * appliedTraction.data[frontWall, -1]
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )
            diagnostics = diagnose(plot=f'vx_s{step}_it{ii+1}.png')

            dv_integral = uw.utils.Integral( fn.math.dot( velocityField - temporaryVelocity,
                                                   velocityField - temporaryVelocity ), mesh )
            dv_norm = np.sqrt(dv_integral.evaluate()[0] / volume)
            v_integral = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )
            v_norm = np.sqrt(v_integral.evaluate()[0] / volume)

            dtau_integral = uw.utils.Integral( fn.math.dot( appliedTraction - temporaryTraction,
                                                            appliedTraction - temporaryTraction ), 
                                               mesh, 'surface', frontWall )
            dtau_norm = np.sqrt(dtau_integral.evaluate()[0] / front_area)
            tau_integral = uw.utils.Integral( fn.math.dot( appliedTraction, appliedTraction ), 
                                              mesh, 'surface', frontWall )
            tau_norm = np.sqrt(tau_integral.evaluate()[0] / front_area)

            expectedTraction.data[frontWall, 0] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                    ((longTermVelocity.data[frontWall, 0] 
                                                    - velocityField.data[frontWall, 0]) / resist_length
                                                    + previousTraction.data[frontWall, 0]
                                                    / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))
            expectedTraction.data[frontWall, -1] = resistShearModulus.evaluate(mesh.data[frontWall])[:, 0] * dt * \
                                                    ((longTermVelocity.data[frontWall, -1] 
                                                    - velocityField.data[frontWall, -1]) / resist_length
                                                    + previousTraction.data[frontWall, -1]
                                                    / (shearModulusFn.evaluate(mesh.data[frontWall])[:, 0]*dt))

            residual_integral = uw.utils.Integral( fn.math.dot( expectedTraction - appliedTraction,
                                                                expectedTraction - appliedTraction ), 
                                                   mesh, 'surface', frontWall )
            residual_norm = np.sqrt(residual_integral.evaluate()[0] / front_area)
            traction_integral = uw.utils.Integral( fn.math.dot( expectedTraction, expectedTraction ), 
                                                   mesh, 'surface', frontWall )
            traction_norm = np.sqrt(traction_integral.evaluate()[0] / front_area)
            
            if(uw.mpi.rank == 0):
                msg = f'|dv| (mm/yr), |v| (mm/yr), |dv|/|v|\n'
                msg += f'{v_mm_yr(dv_norm):.6e} {v_mm_yr(v_norm):.6e} {dv_norm/v_norm:.6e}\n'
                msg += f'|dtau| (Pa), |tau| (Pa), |dtau|/|tau|\n'
                msg += f'{tau_Pa(dtau_norm):.6e} {tau_Pa(tau_norm):.6e} {dtau_norm/tau_norm:.6e}\n'
                msg += f'|Residual| (Pa), |Expectation| (Pa), |R|/|E|\n'
                msg += f'{tau_Pa(residual_norm):.6e} {tau_Pa(traction_norm):.6e} {residual_norm/traction_norm:.6e}'
                logger.debug(msg)

            # If misfit < tol, break
            if(residual_norm/traction_norm < residual_tol and ii+1 >= nits_min):
                break
            else:
                ii += 1

        appliedTraction.save(str(data_dir / f'applied_traction_{flag}{step:d}.h5'))
        expectedTraction.save(str(data_dir / f'expected_traction_{flag}{step:d}.h5'))
    
    dt = dt_e*phi_dt

    pt_vars = getGlobalSwarmVar(np.hstack((swarm.data,
                                            materialIndex.data,
                                            slipVelocity.evaluate(swarm),
                                            stressFn_2nd_Invariant.evaluate(swarm))))
    pt_loc = pt_vars[:, :mesh.dim]
    pt_vars = pt_vars[:, mesh.dim:]

    if(uw.mpi.rank == 0):
        pt_material = pt_vars[:, 0]
        arg_shearz = np.where(np.isclose(pt_material, shear_zone.index))[0]
        if(step == 0):
            np.savez(slipv_stress_dir / f'vars_{flag}{step}.npz', loc=pt_loc[arg_shearz], vars=pt_vars[arg_shearz, 1:])
        else:
            np.savez(slipv_stress_dir / f'vars_{flag}{step}_L{real_resist_length:.0e}.npz', loc=pt_loc[arg_shearz], vars=pt_vars[arg_shearz, 1:])

    stress_data = stressFn.evaluate(swarm)
    previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

    if(step == 0):
        velocityField.save(str(velocity_dir / f'velocity_{flag}{step:d}.h5'))
    else:
        velocityField.save(str(velocity_dir / f'velocity_{flag}{step:d}_L{real_resist_length:.0e}.h5'))

    step = step + 1
    time = time + dt


if(uw.mpi.rank == 0):
    logger.debug(f'Completed using {uw.mpi.size} threads\n')
