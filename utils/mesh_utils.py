import numpy as np
from scipy.linalg import solve

def deform_base_func(x, left, right, buffer):
    x = np.array(x, ndmin=1)
    f_x = np.zeros(len(x))
    half_buffer = buffer/2. 
    for i in range(len(x)):
        if(x[i] < left-half_buffer):
            f_x[i] = x[i]
        elif(x[i] < left+half_buffer):
            f_x[i] = buffer/np.pi * np.sin(np.pi/buffer*(x[i]-(left-half_buffer))) + left-half_buffer
        elif(x[i] < right-half_buffer):
            f_x[i] = 2.*left - x[i]
        elif(x[i] < right+half_buffer):
            f_x[i] = -buffer/np.pi * np.sin(np.pi/buffer*(x[i]-(right-half_buffer))) + 2.*left-(right-half_buffer)
        else:
            f_x[i] = x[i] + 2.*(left-right)
    
    return f_x

def left_right_forward_map(left, right, coef):
    f_left = (left*coef+left)/(coef+2.*(left-right)*coef+1.)
    f_right = ((2.*left-right)*coef+right)/(coef+2.*(left-right)*coef+1.)
    return f_left, f_right

def left_right_backward_map(left, right, coef):
    A = np.array([[coef+1.-2.*left*coef, 2.*left*coef],
                  [2.*coef-2.*right*coef, 2.*right*coef-coef+1]])
    b = np.array([[left+left*coef], [right+right*coef]])
    sol = solve(A, b)
    return sol[0, 0], sol[1, 0]

def deform_map(x, xmin, xmax, left, right, buffer, ratio):
    xx = (x - xmin) / (xmax - xmin)
    xleft = (left - xmin) / (xmax - xmin)
    xright = (right - xmin) / (xmax - xmin)
    xbuffer = buffer / (xmax - xmin)
    xcoef = (ratio - 1.) / (ratio + 1.)
    xleft, xright = left_right_backward_map(xleft, xright, xcoef)
    x_deform = xx + deform_base_func(xx, xleft, xright, xbuffer) * xcoef
    xmin_deform = 0. + deform_base_func(0., xleft, xright, xbuffer)[0] * xcoef
    xmax_deform = 1. + deform_base_func(1., xleft, xright, xbuffer)[0] * xcoef
    x_deform = (x_deform - xmin_deform) / xmax_deform
    x_deform = x_deform * (xmax - xmin) + xmin

    return x_deform