from visual import *
import numpy as np
import random
from math import sqrt, sin, cos, tan, atan, radians, pi

''' CLASSES '''

class Euler:
    '''A class for Euler Angles.'''

    SCALE = 90

    def __init__(self, roll, pitch):
        if abs(roll) > Euler.SCALE or abs(pitch) > Euler.SCALE:
            raise ValueError('Invalid Euler Angle, ' + \
                'range should be between -90 and 90 degrees.')

        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    @classmethod
    def fromAngles(cls, rpy):
        '''
        Builds a new Euler Angle instance from a tuple of values.

        Keyword arguments:
        rpy -- a tuple in the format (roll, pitch)
        '''
        return cls(rpy[0], rpy[1])

    ''' Methods '''

    def rotate(self):
        '''
        Returns the new body orientation of our mimsy board with respect
        to the inertial frame given updated roll and pitch.
        '''
        result = Euler.rotation_op(self.roll, self.pitch)
        return (-result[1] * .5, result[0] * .5, 0), \
            float(sqrt(result[0]**2 + result[1]**2))

    ''' Static Methods '''

    @staticmethod
    def rotation_op(_phi, _theta):
        '''
        Given roll and pitch angles (assuming yaw is 0) describing the body
        orientation of our mimsy board with respect to the inertial frame,
        we use the rotation matrix Q_I/B = [
            [ cos(theta), sin(theta) * sin(phi), cos(phi) * sin(theta)]
            [cos(theta), cos(phi), -sin(phi)]
            [-sin(theta), cos(theta) * sin(phi), cos(theta) * cos(phi)]
        ] to return the i and j components of the projection of the k_hat
        vector np.transpose([0, 0, 1]) onto a 2D plane by dotting the
        matrix with the k_hat vector. We simplified the calculation below.

        Keyword arguments:
        _phi -- the roll angle in our inertial reference frame [-90, 90]
        _theta -- the pitch angle in our inertial reference frame [-90, 90]
        '''
        phi, theta = radians(_phi), radians(_theta)
        return -sin(theta), cos(theta) * sin(phi)

class Network:
    '''A Network class.'''

    _id = 0
    Z_OFFSET = 0.5

    def __init__(self, w, h, inputs=[], mapping={}):
        self.w = w
        self.h = h

        self.scene = display(title='Network' + str(Network._id) + 'Visualization', x=0, y=0)
        self.scene.background = (0.5,0.5,0.5)

        cx, cy = self.quadrant_coors(w*h // 2, w)
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
            x0, y0 = self.quadrant_coors(i, w)
            b = box(pos=(x0,y0,0), length=1, height=1, width=0.2, color=(1,1,1))
            self.mimsies.append(b)

        self.inputs = np.array(inputs)
        if inputs == []:
            self.inputs = np.zeros(shape=(w*h, 3))

        self.update(init=True)

        Network._id += 1

    def update(self, inputs=[], init=False):
        '''
        Updates the state of our network.

        Keyword arguments:
        inputs -- a list with elements of format (roll, pitch)
        init -- True if initializing a network, False otherwise
        '''
        if not init:
            self.inputs = np.array(inputs)
            [Network.rm(vec) for vec in self.vecs]

        if self.inputs.shape != (self.w*self.h, 3):
            raise ValueError('Input shape is ' + str(self.inputs.shape) \
                + ', should be (' + str(w*h) + ', 3)')

        self.vecs = []
        self.angles = []
        for i, rpy in enumerate(self.inputs):
            x1, y1 = self.quadrant_coors(i, self.w)
            angle = Euler.fromAngles(rpy)
            self.angles.append(angle)
            proj, norm = angle.rotate()
            v = arrow(pos=(x1,y1,Network.Z_OFFSET), axis=proj, shaftwidth=0.05, \
                color=(0, 0.5*(1 + norm), 0))
            self.mimsies[i].color = (0, 0, 0.5*(1 + norm))
            self.vecs.append(v)

    def quadrant_coors(self, ID):
        '''
        Given some linearized index, returns the (x, y) pair denoting
        the location of the board in the network.

        Keyword arguments:
        ID -- the linearized index
        '''
        x, y = ID % self.w, ID // self.w
        return x, y

    ''' Static Methods '''

    @staticmethod
    def rm(obj):
        '''
        Removes an object from our scene and from memory.

        Keyword arguments:
        obj -- the visual Python object in our scene
        '''
        obj.visible = False
        del obj

    ''' Class Methods '''

    @classmethod
    def initialize(cls, w=3, h=5, inputs=[]):
        if inputs == []:
            low = -90
            high = 90
            test_inputs = [(0, 0) for k in range(w*h)]
            network = cls(w, h, test_inputs)
        else:
            network = cls(w, h, inputs)

        return network
