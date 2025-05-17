import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
from scipy.linalg import expm, norm
import matplotlib.pyplot as plt

#
# Copyright (c) 2020 Flight Dynamics and Control Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

def test_controllerAllinOne():
    # addpath('aux_functions');
    # addpath('test_functions');

    ### Simulation parameters
    t = np.arange(0, 10.01, 0.01)
    N = len(t)


    param = {
        'm': 4,
        'g': 9.81
    }

    ### Controller gains
    k = {
        'x': 10,
        'v': 10,
        'i': 0,
        'c1': 1,
        'sigma': 1,
    }

    ### Initial conditions
    x0 = 0
    v0 = 0

    X0 = np.array([x0, v0, 0])

    ### Numerical integration
    X = odeint(eom, X0, t, args=(k, param), rtol=1e-6, atol=1e-6)

    ### Post processing
    # Create empty arrays to save data
    e, d, f = generate_output_arrays(N)

    # Unpack the outputs of ode45 function
    x = X[:, 0].T
    v = X[:, 1].T
    ei = X[:, 2].T

    for i in range(N):
        des = command(t[i])
        f[i], _, err = h_control(X[i,:], des, k, param)

        # Unpack errors
        e['x'][i] = err['x']
        e['v'][i] = err['v']

        # Unpack desired values
        d['x'][i] = des['x']
        d['v'][i] = des['v']

    # Plot data
    linetype = 'k'
    linewidth = 1
    xlabel_ = 'time (s)'

    plt.figure(2)
    plt.plot(t, e['x'], color=linetype, linewidth=linewidth)
    plt.grid(True)
    plt.xlabel(xlabel_)
    plt.ylabel('e_x')
    
    plt.figure(3)
    plt.plot(t, e['v'], color=linetype, linewidth=linewidth)
    plt.grid(True)
    plt.xlabel(xlabel_)
    plt.ylabel('e_v')

    plt.figure(5)
    plt.plot(t, ei * k['i'], color=linetype, linewidth=linewidth)
    plt.grid(True)
    plt.xlabel(xlabel_)
    plt.ylabel('e_i')
    
    plt.figure(6)
    plt.subplot(211)
    plt.plot(t, x,  color=linetype, linewidth=linewidth)
    plt.plot(t, d['x'],  color='r', linewidth=linewidth)
    plt.grid(True)
    plt.xlabel(xlabel_)
    plt.ylabel('x')
    plt.legend(['x', 'd_x'])
    plt.subplot(212)
    plt.plot(t, v,  color=linetype, linewidth=linewidth)
    plt.plot(t, d['v'],  color='r', linewidth=linewidth)
    plt.grid(True)
    plt.xlabel(xlabel_)
    plt.ylabel('v')
    plt.legend(['v', 'd_v'])
    
    print('')

def eom(X, t, k, param):
    m = param['m']

    _, v, _ = split_to_states(X)

    desired = command(t)
    f, ei_dot, _ = h_control(X, desired, k, param)

    xdot = v
    vdot = -param['g'] + f / m

    return np.array([xdot, vdot, ei_dot])

def command(t):
    desired = command_sin(t)
    # desired = command_point(t)
    return desired

def h_control(X, desired, k, param):
    """[f, M, ei_dot, eI_dot, error_, calculated] = h_control(X, desired, k, param)

      Inputs:
       X: (24x1 matrix) states of the system (x, v, ei)
       desired: (struct) desired states
       k: (struct) control gains
       param: (struct) parameters such as m, g in a struct
    
      Outputs:
        f: (scalar) required motor force
        ei_dot: (3x1 matrix) position integral change rate
    """

    # Unpack states
    x, v, ei = split_to_states(X)

    sigma = k['sigma']
    c1 = k['c1']
    m = param['m']
    g = param['g']

    error_ = {}
    error_['x'] = desired['x'] - x                                               
    error_['v'] = desired['v'] - v   
                                                
    A =   k['x'] * error_['x'] \
        + k['v'] * error_['v'] \
        + k['i'] * np.clip(ei, -sigma, sigma) \
        + m * desired['x_2dot'] \
        + m * g

    ei_dot = error_['v'] + c1 * error_['x']                                       # (13)
    f = A

    return f, ei_dot, error_

def split_to_states(X):
    x = X[0]
    v = X[1]
    ei = X[2]
    return x, v, ei

def generate_output_arrays(N):
    error_ = {
        'x': np.zeros(N),
        'v': np.zeros(N),
    }

    desired = {
        'x': np.zeros(N),
        'v': np.zeros(N),
    }

    f = np.zeros(N)

    return error_, desired, f

def command_sin(t, w=2):
    height = 0

    desired = {
        'x': sin(w * t),
        'v': w * cos(w * t),
        'x_2dot': -w**2 * sin(w * t),
    }

    w = 2 * np.pi / 10
    desired['b1'] = np.array([np.cos(w * t), np.sin(w * t), 0])
    desired['b1_dot'] = w * np.array([-np.sin(w * t), np.cos(w * t), 0])
    desired['b1_2dot'] = w**2 * np.array([-np.cos(w * t), -np.sin(w * t), 0])

    return desired

def command_point(t):
    height = 0

    desired = {
        'x': 20,
        'v': 0,
        'x_2dot': 0,
    }

    return desired

if __name__ == "__main__":
    test_controllerAllinOne()
    plt.show()
    i=1