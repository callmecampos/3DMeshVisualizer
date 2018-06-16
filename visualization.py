import numpy as np
import random
from math import sqrt, sin, cos, tan, radians, pi
from mayavi.mlab import *

''' CLASSES '''

class Euler:
    '''A class for Euler Angles.'''

    SCALE = 90

    def __init__(self, roll, pitch, yaw=0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    @classmethod
    def fromAngles(cls, rpy):
        return cls(rpy[0], rpy[1], rpy[2])

    ''' Methods '''

    def rotate(self):
        result = Euler.rotationOp(self.roll, self.pitch, self.yaw)
        return (result[0] * .5, result[1] * .5, result[2] * .5), \
            float(sqrt(result[0]**2 + result[1]**2 + result[2]**2))

    ''' Static Methods '''

    @staticmethod
    def rotationOp(_phi, _theta, _psi):
        psi, theta, phi = radians(_psi), radians(_theta), -radians(_phi)
        k_hat = np.transpose(np.array([0, 0, 1]))
        Q_BI = np.zeros(shape=(3, 3))
        Q_BI[0] = [cos(theta) * cos(psi),
                    cos(psi) * sin(theta) * sin(phi) - cos(phi) * sin(psi),
                    sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)]
        Q_BI[1] = [cos(theta) * cos(psi),
                    cos(phi) * cos(psi) + sin(theta) * sin(phi) * sin(psi),
                    cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)]
        Q_BI[2] = [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]

        '''print('sin(psi), cos(psi) = ' + str(sin(psi)) + ', ' + str(cos(psi)))
        print('sin(theta), cos(theta) = ' + str(sin(theta)) + ', ' + str(cos(theta)))
        print('sin(phi), cos(phi) = ' + str(sin(phi)) + ', ' + str(cos(phi)))
        print(Q_BI, k_hat)'''

        return np.dot(Q_BI, k_hat)

    @staticmethod
    def norm(a):
        '''Returns the "magnitude" of the roll, pitch, or yaw, normalized to 1.

        Keyword args:
        a -- the roll, pitch, or yaw, with an inclusive range of -SCALE to SCALE (degrees).
        '''
        a = max(-SCALE,a)
        a = min(SCALE,a)
        return a / abs(SCALE)

class Network:
    '''A Network class.'''

    _id = 0

    def __init__(self, w, h, inputs=[]):
        self.w = w
        self.h = h

        cx, cy = Network.quadrant_coors(w*h // 2, w)
        cz = 0 # FIXME??
        self.center = (cx, cy, cz)

        self.mimsies = []
        for i in range(w*h):
            x0, y0 = Network.quadrant_coors(i, w)
            b = (x0, y0) # FIXME
            self.mimsies.append(b)

        self.inputs = np.array(inputs)
        if inputs == []:
            self.inputs = np.zeros(shape=(w*h, 3))

        self.update()

        Network._id += 1

    @classmethod
    def initialize(cls, w=3, h=5, inputs=[]):
        if inputs == []:
            test_inputs = [[0, 0, 0] for k in range(w*h)]
            network = cls(w, h, test_inputs)
        else:
            network = cls(w, h, inputs)

        return network

    def update(self, inputs):
        # update state
        self.inputs = np.array(inputs)

        if self.inputs.shape != (self.w*self.h, 3):
            raise ValueError('Input shape is ' + str(self.inputs.shape) + \
                ', should be (' + str(self.w*self.h) + ', 3)')

        for tup in inputs:
            for val in tup:
                if val < -Euler.SCALE or val > Euler.SCALE:
                    raise ValueError('Invalid Euler Angle, ' + \
                        'range should be between -90 and 90 degrees.')

        self.angles = [Euler.fromAngles(rpy) for rpy in self.inputs]
        self.makeVectors()

    def display(self):
        quiver3d(*self.vecs)
        show()

    def makeVectors(self):
        x = np.zeros(shape=(len(self.angles),))
        y = np.zeros(shape=(len(self.angles),))
        z = np.zeros(shape=(len(self.angles),))
        u = np.zeros(shape=(len(self.angles),))
        v = np.zeros(shape=(len(self.angles),))
        w = np.zeros(shape=(len(self.angles),))
        for i, angle in enumerate(self.angles):
            x[i], y[i] = Network.real_quadrant_coors(i, self.w) # FIXME??
            z[i] = 0
            proj, norm = angle.rotate()
            u[i], v[i], w[i] = proj
        self.vecs = (x, y, z, u, v, w)

    ''' Static Methods '''

    @staticmethod
    def quadrant_coors(ID, w):
        x, y = ID % w, ID // w
        return x, y

    @staticmethod
    def real_quadrant_coors(ID, w):
        x, y = Network.quadrant_coors(ID, w)
        return x + .5, y + .5

''' VISUALIZATION '''

network = Network.initialize()
network.display()
