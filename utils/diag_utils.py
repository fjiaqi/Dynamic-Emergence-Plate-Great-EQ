import numpy as np
from scipy.interpolate import LinearNDInterpolator
import sys
sys.path.append('./model/')
from references import *
from domain import *
from materials import *

def l_m(length):
    return length*reference_length

def l_km(length):
    return length*reference_length/1e3

def t_s(time):
    return time*reference_time

def t_yr(time):
    return time*reference_time/s_per_year

def v_m_s(velocity):
    return velocity*reference_velocity

def v_mm_yr(velocity):
    return velocity*reference_velocity*s_per_year*1e3

def v_cm_yr(velocity):
    return velocity*reference_velocity*s_per_year*1e2

def tau_Pa(stress):
    return stress*reference_stress

def tau_MPa(stress):
    return stress*reference_stress/1e6

def logeta_Pa_s(viscosity):
    return np.log10(viscosity*reference_viscosity)

def logeps_s(strain_rate):
    return np.log10(strain_rate/reference_time)

def T_K(temperature):
    return reference_T0+temperature*(reference_T1-reference_T0)

def T_C(temperature):
    return reference_T0+temperature*(reference_T1-reference_T0)-273.15


def second_invariant(A, dim=2):
    if(dim == 2 or A.shape[1] == 3):
        res = np.sqrt(A[:, 0]**2/2. + A[:, 1]**2/2. + A[:, 2]**2)
    elif(dim == 3 or A.shape[1] == 6):
        res = np.sqrt(A[:, 0]**2/2. + A[:, 1]**2/2. + A[:, 2]**2/2. +
                      A[:, 3]**2 + A[:, 4]**2 + A[:, 5]**2)
    else:
        raise ValueError('dim must be 2 or 3')
    return res

def SHmax(A, dim=2):
    if(dim == 2):
        tau11 = A[:, 0]
        tau22 = A[:, 1]
        tau12 = A[:, 2]
        l1 = (tau11 + tau22 - np.sqrt((tau11-tau22)**2 + 4*tau12**2)) / 2.
        angle1 = np.arctan2(tau11-l1, -tau12)
    elif(dim == 3):
        # TODO: check this
        angle1 = np.zeros(A.shape[0])
    else:
        raise ValueError('dim must be 2 or 3')
    return angle1


def plate_quantity(var, x, xmin, xmax, **kwargs):
    arg_p = np.where((x>xmin) & (x<xmax))[0]
    res = np.array([np.average(var[arg_p])])
    if(kwargs.get('return_std', False)):
        res = np.append(res, np.std(var[arg_p]))
    if(kwargs.get('return_min', False)):
        res = np.append(res, np.min(var[arg_p]))
    if(kwargs.get('return_max', False)):
        res = np.append(res, np.max(var[arg_p]))
    if(kwargs.get('return_rms', False)):
        res = np.append(res, np.sqrt(np.average(var[arg_p]**2)))
    return res

def shear_zone_quantity(var, material, **kwargs):
    arg_sz = np.where(material == shear_zone.index)[0]
    res = np.array([np.average(var[arg_sz])])
    if(kwargs.get('return_std', False)):
        res = np.append(res, np.std(var[arg_sz]))
    if(kwargs.get('return_min', False)):
        res = np.append(res, np.min(var[arg_sz]))
    if(kwargs.get('return_max', False)):
        res = np.append(res, np.max(var[arg_sz]))
    if(kwargs.get('return_rms', False)):
        res = np.append(res, np.sqrt(np.average(var[arg_sz]**2)))
    return res

def interpolated_field_func(loc, var):
    return LinearNDInterpolator(loc, var)