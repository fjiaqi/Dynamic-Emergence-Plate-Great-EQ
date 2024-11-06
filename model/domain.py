from references import *

age_west_plate = 50e6 * s_per_year / reference_time
age_east_plate = 80e6 * s_per_year / reference_time
age_max_cut = 80e6 * s_per_year / reference_time
vel_west_plate = 5e-2 / s_per_year / reference_velocity

shear_zone_width = 6e3 / reference_length
ridge_width = 40e3 / reference_length
ridge_loc = -age_west_plate*vel_west_plate
slab_top_max_depth = 400e3 / reference_length
west_crust_depth = 6e3 / reference_length
east_crust_depth = 20e3 / reference_length

domain_width_west = -ridge_loc + ridge_width
domain_width_east = 1500e3 / reference_length
domain_width = domain_width_east + domain_width_west
domain_depth = 660e3 / reference_length
lmantle_top_depth = 660e3 / reference_length
air_depth = 0. / reference_length
domain_length = 5e3 / reference_length

xres = 1024
zres = 256
yres = 0
restart_step = -100