from visual import *
import numpy as np
import random
from math import sqrt, sin, cos, tan, atan, radians, pi

''' CLASSES '''

class Euler:
    '''A class for Euler Angles.'''

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
        return (-result[1] * .5, result[0] * .5, 0), \
            float(sqrt(result[0]**2 + result[1]**2 + result[2]**2))

    ''' Static Methods '''

    @staticmethod
    def rotationOp(_phi, _theta, _psi):
        psi, theta, phi = radians(_psi), radians(_theta), radians(_phi)
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
        a -- the roll, pitch, or yaw, with an inclusive range of -90 to 90 (degrees).
        '''
        a = max(-90,a)
        a = min(90,a)
        return a / 90.0

class Network:
    '''A Network class.'''

    _id = 0
    Z_OFFSET = 0.5

    MIMSY = { "def01" : 0 }

    def __init__(self, w, h, inputs=[]):
        self.w = w
        self.h = h

        self.scene = display(title='Network' + str(Network._id) + 'Visualization', x=0, y=0)
        self.scene.background = (0.5,0.5,0.5)

        cx, cy = Network.quadrant_coors(w*h // 2, w)
        cz = self.scene.center.z
        self.center = (cx, cy, cz)
        self.scene.center = (cx,cy,cz)
        self.scene.forward = (0,0,-1)
        self.scene.range = 8

        self.scene.lights = [vector(1,0,0), vector(0, 1, 0), vector(0, 0, 1), \
            vector(-1,0,0), vector(0, -1, 0), vector(0, 0, -1)]
        self.scene.ambient = 0

        self.mimsies = []
        for i in range(w*h):
            x0, y0 = Network.quadrant_coors(i, w)
            b = box(pos=(x0,y0,0), length=1, height=1, width=0.2, color=(1,1,1))
            self.mimsies.append(b)

        self.inputs = np.array(inputs)
        if inputs == []:
            self.inputs = np.zeros(shape=(w*h, 3))

        self.update(init=True)

        Network._id += 1

    def update(self, inputs=[], init=False):
        # update state
        if not init:
            self.inputs = np.array(inputs)
            [Network.rm(vec) for vec in self.vecs]

        if self.inputs.shape != (self.w*self.h, 3):
            raise ValueError('Input shape is ' + str(self.inputs.shape) \
                + ', should be (' + str(w*h) + ', 3)')

        for tup in inputs:
            for val in tup:
                if val < -90 or val > 90:
                    raise ValueError('Invalid Euler Angle, ' + \
                        'range should be between -90 and 90 degrees.')

        self.vecs = []
        self.angles = []
        for i, rpy in enumerate(self.inputs):
            x1, y1 = Network.quadrant_coors(i, self.w)
            angle = Euler.fromAngles(rpy)
            self.angles.append(angle)
            proj, norm = angle.rotate()
            v = arrow(pos=(x1,y1,Network.Z_OFFSET), axis=proj, shaftwidth=0.05, \
                color=(0, 0.5*(1 + norm), 0))
            self.mimsies[i].color = (0, 0, 0.5*(1 + norm))
            self.vecs.append(v)

    ''' Static Methods '''

    @staticmethod
    def rm(obj):
        obj.visible = False
        del obj

    @staticmethod
    def quadrant_coors(ID, n):
        x, y = ID % n, ID // n
        return x, y

    @staticmethod
    def real_quadrant_coors(ID, n):
        x, y = Network.quadrant_coors(ID, n)
        return x + .5, y + .5

    ''' Class Methods '''

    @classmethod
    def initialize(cls, w=3, h=5, inputs=[]):
        if inputs == []:
            low = -90
            high = 90
            test_inputs = [[0, 0, 0] for k in range(w*h)]
            network = cls(w, h, test_inputs)
        else:
            network = cls(w, h, inputs)

        return network
