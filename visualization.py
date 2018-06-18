from visual import *
import numpy as np
import random
from math import sqrt, sin, cos, tan, atan, radians, pi
from bidict import bidict

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

    def __init__(self, w, h, mapping):
        self.w = w
        self.h = h

        self.mapping = mapping

        self.scene = display(title='Network' + str(Network._id) + 'Visualization', x=0, y=0)
        self.scene.background = (0.5,0.5,0.5)

        cx, cy = self.quadrant_coors(w*h // 2)
        cz = self.scene.center.z
        self.center = (cx, cy, cz)
        self.scene.center = (cx,cy,cz)
        self.scene.forward = (0,0,-1)
        self.scene.range = max(w, h)

        self.scene.lights = [vector(1,0,0), vector(0, 1, 0), vector(0, 0, 1), \
            vector(-1,0,0), vector(0, -1, 0), vector(0, 0, -1)]
        self.scene.ambient = 0

        self.mimsies = []
        for i in range(w*h):
            x0, y0 = self.quadrant_coors(i)
            b = box(pos=(x0,y0,0), length=1, height=1, width=0.2, color=(1,1,1))
            self.mimsies.append(b)

        self.inputs = np.array([(0, 0) for k in range(w*h)])
        self.initGUI()

        Network._id += 1

    def update(self, data, addr):
        '''
        Updates the state of our network.

        Keyword arguments:
        inputs -- a list with elements of format (roll, pitch)
        init -- True if initializing a network, False otherwise
        '''

        # get vec from mapping
        # set new params

    def initGUI(self):
        '''
        Initializes the GUI for our network.

        Keyword arguments:
        inputs -- a list with elements of format (roll, pitch)
        '''

        self.checkInput()

        self.vecs = []
        self.angles = []
        for i, rpy in enumerate(self.inputs):
            x1, y1 = self.quadrant_coors(i)
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

    def checkInput(self):
        '''
        Checks the shape of our input, throws a ValueError if formatted badly.
        '''
        if self.inputs.shape != (self.w*self.h, 2):
            raise ValueError('Bad Input: Input shape is ' + str(self.inputs.shape) \
                + ', should be (' + str(self.w*self.h) + ', 2)')


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
    def initialize(cls, filename):
        '''
        Parses a text file for the network format and initializes a new network.

        Keyword arguments:
        filename -- the name of the file to parse (ideally text)
        '''
        f = open(filename, "r")
        content = f.readlines()

        h = len(content)
        w = 0
        set = False

        mapping = bidict({})
        for i, line in enumerate(content):
            keys = line.replace(" ", "").split(',')
            if not set:
                w = len(keys)
                set = True
            [mapping.put(i*len(keys)+j, key) for j, key in enumerate(keys)]

        return cls(w, h, mapping=mapping)
