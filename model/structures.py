import numpy as np
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d
import shapely.geometry as shg
from shapely import ops
from references import *
from domain import *

z_arr = np.linspace(-domain_depth*1.2, 0, 501)
t_arr = np.linspace(0, np.maximum(age_west_plate, age_east_plate)*1.2, 101)
T_arr = np.zeros((len(z_arr), len(t_arr)))
for i in range(len(t_arr)):
    if(t_arr[i] > 0):
        # T_arr[:, i] = erf(-z_arr*reference_length/2./np.sqrt(reference_diffusivity*tz * reference_time))
        T_arr[:, i] = erf(-z_arr/2./np.sqrt(t_arr[i]))
    else:
        T_arr[:, i] = 1.
        T_arr[-1, i] = 0.

zz_plate, tt_plate = np.meshgrid(z_arr, t_arr)
plate_temp_func = LinearNDInterpolator((zz_plate.ravel(), tt_plate.ravel()), T_arr.T.ravel(), fill_value=1)


surface_coords = np.array([[-domain_width_west, 0.], [domain_width_east, 0.]])
surface_line = shg.LineString(surface_coords)
top_coords = np.array([[-domain_width_west, air_depth], [domain_width_east, air_depth]])
top_line = shg.LineString(top_coords)
bottom_coords = np.array([[-domain_width_west, -domain_depth], [domain_width_east, -domain_depth]])
bottom_line = shg.LineString(bottom_coords)
west_wall_coords = np.array([[-domain_width_west, -domain_depth], [-domain_width_west, air_depth]])
west_wall_line = shg.LineString(west_wall_coords)
east_wall_coords = np.array([[domain_width_east, -domain_depth], [domain_width_east, air_depth]])
east_wall_line = shg.LineString(east_wall_coords)

n_interp = 101
west_lithos_bottom_coords = np.zeros((n_interp, 2))
west_lithos_bottom_coords[:, 0] = np.linspace(ridge_loc, 0, n_interp)
lithos_threshold = 0.9
z_arr_new = np.linspace(z_arr[np.where(T_arr[:, -1] <= lithos_threshold)[0][0]], 0, 1001)
for i in range(1, n_interp):
    xp = west_lithos_bottom_coords[i, 0]
    tz = plate_temp_func((z_arr_new, np.repeat((xp-ridge_loc)/vel_west_plate, len(z_arr_new))))
    west_lithos_bottom_coords[i, 1] = z_arr_new[np.where(tz <= lithos_threshold)[0][0]]
west_lithos_bottom_xzfunc_tmp = interp1d(west_lithos_bottom_coords[:, 0], west_lithos_bottom_coords[:, 1], 
                                       fill_value='extrapolate')

slab_top_coords = np.genfromtxt('./geodata/slab_top_data2.txt')[::-1, :]*1e3/reference_length
slab_top_coords[:, 1] = slab_top_coords[:, 1] - slab_top_coords[:, 1].max()
slab_top_down_slope = np.polyfit(slab_top_coords[-5:, 0], slab_top_coords[-5:, 1], 1)[0]
if(-slab_top_max_depth < slab_top_coords[-1, 1]):
    slab_top_coords = np.vstack((slab_top_coords, 
                                 np.array([slab_top_coords[-1, 0]-(slab_top_coords[-1, 1]+slab_top_max_depth)/slab_top_down_slope, 
                                           -slab_top_max_depth])))
slab_top_line = shg.LineString(slab_top_coords)
slab_top_xzfunc = interp1d(slab_top_coords[:, 0], slab_top_coords[:, 1], fill_value='extrapolate')
slab_top_zxfunc = interp1d(slab_top_coords[:, 1], slab_top_coords[:, 0], fill_value='extrapolate')

shear_zone_bottom_line = slab_top_line.offset_curve(-shear_zone_width)
shear_zone_bottom_coords = np.array(shear_zone_bottom_line.coords)
shear_zone_bottom_up_slope = np.polyfit(shear_zone_bottom_coords[:2, 0], shear_zone_bottom_coords[:2, 1], 1)[0]
shear_zone_bottom_coords = np.vstack((np.array([shear_zone_bottom_coords[0, 0]
                                                -shear_zone_bottom_coords[0, 1]/shear_zone_bottom_up_slope, 0]), 
                                      shear_zone_bottom_coords))
shear_zone_bottom_line = shg.LineString(shear_zone_bottom_coords)
shear_zone_bottom_xzfunc = interp1d(shear_zone_bottom_coords[:, 0], shear_zone_bottom_coords[:, 1], fill_value='extrapolate')
shear_zone_bottom_zxfunc = interp1d(shear_zone_bottom_coords[:, 1], shear_zone_bottom_coords[:, 0], fill_value='extrapolate')

slab_bottom_line = slab_top_line.offset_curve(west_lithos_bottom_coords[-1, 1])
slab_bottom_coords = np.array(slab_bottom_line.coords)
slab_bottom_xzfunc_tmp = interp1d(slab_bottom_coords[:, 0], slab_bottom_coords[:, 1], fill_value='extrapolate')

n_interp = 9
joint_coords = np.zeros((n_interp, 2))
joint_coords[:, 0] = np.linspace(slab_bottom_coords[:, 0].min(), 0, n_interp+2)[1:-1]
for i in range(n_interp):
    xp = joint_coords[i, 0]
    joint_coords[i, 1] = (slab_bottom_xzfunc_tmp(joint_coords[i, 0]) * (xp-slab_bottom_coords[:, 0].min())
                          + west_lithos_bottom_xzfunc_tmp(joint_coords[i, 0]) * (0-xp)) / (0-slab_bottom_coords[:, 0].min())

west_lithos_bottom_coords = west_lithos_bottom_coords[west_lithos_bottom_coords[:, 0] <= slab_bottom_coords[:, 0].min()]
slab_bottom_coords = slab_bottom_coords[slab_bottom_coords[:, 0] >= 0]
west_lithos_bottom_no_slab_coords = np.vstack((west_lithos_bottom_coords, joint_coords))
west_lithos_bottom_coords = np.vstack((west_lithos_bottom_coords, joint_coords, slab_bottom_coords))
west_lithos_bottom_line = shg.LineString(west_lithos_bottom_coords)
west_lithos_bottom_xzfunc = interp1d(west_lithos_bottom_coords[:, 0], west_lithos_bottom_coords[:, 1], fill_value='extrapolate')

slab_front_coords = np.array([[west_lithos_bottom_coords[-1, 0], west_lithos_bottom_coords[-1, 1]],
                              [slab_top_coords[-1, 0], slab_top_coords[-1, 1]]])
slab_front_slope_rad = np.arctan((slab_front_coords[1, 1] - slab_front_coords[0, 1])
                                 / (slab_front_coords[1, 0] - slab_front_coords[0, 0]))
slab_front_line = shg.LineString(slab_front_coords)
slab_front_xzfunc = interp1d(slab_front_coords[:, 0], slab_front_coords[:, 1], fill_value='extrapolate')

east_lithos_bottom_coords = np.zeros((2, 2))
tz = plate_temp_func((z_arr_new, np.repeat(age_east_plate, len(z_arr_new))))
east_lithos_bottom_coords[:, 0] = np.array([0, domain_width_east])
east_lithos_bottom_coords[:, 1] = z_arr_new[np.where(tz <= lithos_threshold)[0][0]]
east_lithos_bottom_line = shg.LineString(east_lithos_bottom_coords)
pt_tmp = east_lithos_bottom_line.intersection(slab_top_line)
if(pt_tmp.intersects(east_lithos_bottom_line)):
    east_lithos_bottom_line = ops.split(east_lithos_bottom_line, pt_tmp).geoms[-1]
else:
    east_lithos_bottom_line = ops.split(east_lithos_bottom_line, 
                                        pt_tmp.buffer(pt_tmp.distance(east_lithos_bottom_line))).geoms[-1]
east_lithos_bottom_coords = np.array(east_lithos_bottom_line.coords)
east_lithos_bottom_xzfunc = interp1d(east_lithos_bottom_coords[:, 0], east_lithos_bottom_coords[:, 1], 
                                   fill_value='extrapolate')


x_arr_min = 3.*z_arr[np.where(T_arr[:, -1] > 0.95)[0][-1]]
x_arr_max = -2.*z_arr[np.where(T_arr[:, -1] > 0.95)[0][-1]]
x_arr = np.linspace(x_arr_min, x_arr_max, 501)
u0 = -1. + plate_temp_func((x_arr, np.repeat(age_west_plate, len(x_arr))))
u0[x_arr > 0] = -1.

t_arr = np.linspace(0, slab_top_line.length/vel_west_plate*1.2, 101)
u_arr = np.zeros((len(x_arr), len(t_arr)))

kappa = 1.

u_arr[:, 0] = 1+u0
for i in range(1, len(t_arr)):
    pt_tmp = slab_top_line.interpolate(t_arr[i]*vel_west_plate)
    u_east = -1.+plate_temp_func([pt_tmp.xy[1][0], age_east_plate])
    u0[x_arr > 0] = u_east
    ug = gaussian_filter1d(u0, np.sqrt(2.*kappa*t_arr[i])/(x_arr[1]-x_arr[0]), axis=0)
    u_arr[:, i] = 1+ug

t_arr[0] = -(t_arr[1]-t_arr[0])*1e-3 # Avoid interpolation error for small time

xx_slab, tt_slab = np.meshgrid(x_arr, t_arr)
slab_temp_func = LinearNDInterpolator((xx_slab.ravel(), tt_slab.ravel()), u_arr.T.ravel(), fill_value=1)