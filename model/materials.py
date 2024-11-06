from references import *

class Material:
    def __init__(self, index, label, 
                 density, diffusivity, alpha,
                 max_viscosity, min_viscosity, 
                 diff_E, diff_V, diff_A,
                 disl_n, disl_E, disl_V, disl_A,
                 shear_modulus_high, shear_modulus_low,
                 cohesion, max_yield_stress):
        self.index = index
        self.label = label
        self.density = density / reference_density
        self.diffusivity = diffusivity / reference_diffusivity
        self.alpha = alpha
        self.max_viscosity = max_viscosity / reference_viscosity
        self.min_viscosity = min_viscosity / reference_viscosity
        self.diff_E = diff_E
        self.diff_V = diff_V
        self.diff_A = diff_A / reference_viscosity
        self.disl_n = disl_n
        self.disl_E = disl_E
        self.disl_V = disl_V
        self.disl_A = disl_A / (reference_viscosity/reference_time**(1.-1./disl_n))
        self.shear_modulus_high = shear_modulus_high / reference_stress
        self.shear_modulus_low = shear_modulus_low / reference_stress
        self.cohesion = cohesion / reference_stress
        self.max_yield_stress = max_yield_stress / reference_stress


air_dict        = { 'density': 1000., 'diffusivity': 1e-5, 'alpha': 0.,
                    'max_viscosity': 1e18, 'min_viscosity': 1e18,
                    'diff_E': 0., 'diff_V': 0., 'diff_A': 2e18,
                    'disl_n': 1., 'disl_E': 0., 'disl_V': 0., 'disl_A': 2e18,
                    'shear_modulus_high': 750e9, 'shear_modulus_low': 750e9,
                    'cohesion': 500e6, 'max_yield_stress': 500e6 }

mantle_dict     = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    # 'disl_n': 3.5, 'disl_E': 540e3, 'disl_V': 12.5e-6, 'disl_A': 1.4485e4,
                    'disl_n': 3.5, 'disl_E': 532e3, 'disl_V': 11e-6, 'disl_A': 2.7695e4,
                    'shear_modulus_high': 60e9, 'shear_modulus_low': 60e9,
                    'cohesion': 100e6, 'max_yield_stress': 200e6 }

shear_zone_dict = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    # 'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'diff_E': 0e3, 'diff_V': 0e-6, 'diff_A': 1e24,
                    # 'disl_n': 2.6, 'disl_E': 230e3, 'disl_V': 4e-6, 'disl_A': 1.6623e6,
                    'disl_n': 2.3, 'disl_E': 154e3, 'disl_V': 0e-6, 'disl_A': 1.6538e7,
                    'shear_modulus_high': 30e9, 'shear_modulus_low': 30e9,
                    'cohesion': 10e6, 'max_yield_stress': 200e6 }

crust_dict      = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    # 'diff_E': 0e3, 'diff_V': 0e-6, 'diff_A': 1e24,
                    'disl_n': 3.5, 'disl_E': 532e3, 'disl_V': 11e-6, 'disl_A': 2.7695e4,
                    # 'disl_n': 2.3, 'disl_E': 154e3, 'disl_V': 4e-6, 'disl_A': 1.6538e7,
                    'shear_modulus_high': 30e9, 'shear_modulus_low': 30e9,
                    'cohesion': 100e6, 'max_yield_stress': 200e6 }

lmantle_dict    = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 2.5e-6, 'diff_A': 1.7241e12,
                    'disl_n': 1.0, 'disl_E': 0e3, 'disl_V': 0e-6, 'disl_A': 1e24,
                    'shear_modulus_high': 60e9, 'shear_modulus_low': 60e9,
                    'cohesion': 100e6, 'max_yield_stress': 200e6 }

air        = Material(index=0, label='air', **air_dict)
west_plate = Material(index=1, label='west_plate', **mantle_dict)
east_plate = Material(index=2, label='east_plate', **mantle_dict)
shear_zone = Material(index=3, label='shear_zone', **shear_zone_dict)
mantle     = Material(index=4, label='mantle', **mantle_dict)
west_crust = Material(index=5, label='west_crust', **crust_dict)
east_crust = Material(index=6, label='east_crust', **crust_dict)
lmantle    = Material(index=7, label='lmantle', **mantle_dict)