reference_length = 1.0e6
reference_viscosity = 1.0e19
gravity = 10.
reference_density = 3300.
reference_expansivity = 3e-5
reference_diffusivity = 1e-6
reference_T0 = 300.
reference_T1 = 1700.
reference_time = reference_length**2/reference_diffusivity
reference_velocity = reference_length/reference_time
reference_stress = reference_viscosity/reference_time
s_per_year = 3600.*24.*365.25
R_const = 8.314

Ra = (reference_expansivity*reference_density*gravity*(reference_T1-reference_T0)*reference_length**3)/(reference_diffusivity*reference_viscosity)
Rb = (reference_density*gravity*reference_length**3)/(reference_diffusivity*reference_viscosity)