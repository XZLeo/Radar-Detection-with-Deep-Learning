from math import cos, sin, pi
from numpy import array, ndarray, dot
from numpy.random import randint, rand


def rotate30(snip:ndarray):
    '''
    rotate each point in the snippet around the center of the scene, i.e., 
    x_cc = 50 y_cc=100 (in ego-vehicle coordinate)
    randomly rotated by multiples of 30 degree for data augmentation 
    '''
    # generate random integer
    theta = randint(1, 12, size=1) * pi / 6
    # rotation matrix
    rotat = array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    #print('rotat', rotat.shape, rotat)
    # coordinates
    xy = array([snip['x_cc']-50, snip['y_cc']]).transpose() 
    # rotate
    rotated_xy = dot(xy, rotat) + array([50, 0])
    # unpack
    snip['x_cc'] = rotated_xy[:, 0] 
    snip['y_cc'] = rotated_xy[:, 1]
    return snip, theta*6/pi


def rotate(snip:ndarray):
    '''
    rotate each point in the snippet around the center of the scene, i.e., 
    x_cc = 50 y_cc=100 (in ego-vehicle coordinate)
    randomly rotated by multiples of 30 degree for data augmentation 
    '''
    # generate random integer
    rnd_seed = rand(1)
    theta = 2 * pi * rnd_seed 
    # rotation matrix
    rotat = array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    #print('rotat', rotat.shape, rotat)
    # coordinates
    xy = array([snip['x_cc']-50, snip['y_cc']]).transpose() 
    # rotate
    rotated_xy = dot(xy, rotat) + array([50, 0])
    # unpack
    snip['x_cc'] = rotated_xy[:, 0] 
    snip['y_cc'] = rotated_xy[:, 1]
    return snip, rnd_seed*360


def mirror(snip:ndarray, mode):
    '''
    data augmentation by flip the snippet horizontally
    '''
    if mode == 'leftRight':
        # generate random integer
        m = randint(0, 2, size=1)
        
        if m == 1:
            snip['y_cc'] = -snip['y_cc']
    elif mode == 'upDown':
        # generate random integer
        m = randint(0, 2, size=1)
        
        if m == 1:
            snip['x_cc'] = 100 - snip['x_cc']
    elif mode == 'mixture':
        m = randint(0, 4, size=1)
        
        if m == 0:
            snip['y_cc'] = -snip['y_cc']
        elif m == 1:
            snip['x_cc'] = 100 - snip['x_cc']
        elif m == 2:
            snip['y_cc'] = -snip['y_cc']
            snip['x_cc'] = 100 - snip['x_cc']
        else:
            pass
    else:
        print('There is no such mode for mirror. Do nothing.')
    return snip
    

    
