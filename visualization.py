from visual import *
import matplotlib.pyplot as plt
import numpy as np
import random, re, csv
from math import sqrt, sin, cos, tan, atan, radians, degrees, pi
from bidict import bidict

''' CLASSES '''

class Euler:
    '''A class for Euler Angles.'''
    MAX = 90
    SCALE = 1

    def __init__(self, roll, pitch):
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
        return (-result[1] * Euler.SCALE, result[0] * Euler.SCALE, 0), \
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
    DIMENSION = 2
    Z_OFFSET = 0.5

    def __init__(self, w, h, mapping, testing):
        self.w = w
        self.h = h

        self.mapping = mapping

        if testing:
            Euler.SCALE = 0.5

        self.scene = display(title='Network' + str(Network._id) + 'Visualization', x=0, y=0)
        self.scene.background = (0.5,0.5,0.5)

        cx, cy = float(w) / 2 - .5, float(h) / 2 - .5
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
            b = box(pos=(x0,y0,0), length=1, height=1, width=0.2, color=(1,0,0))
            self.mimsies.append(b)

        self.inputs = np.array([(0, 0) for k in range(w*h)])
        self.initGUI()

        print(self.mapping)
        print(len(self.vecs))

        Network._id += 1

    def __len__(self):
        return self.w * self.h

    def update(self, data, addr):
        '''
        Updates the state of our network.

        Keyword arguments:
        data -- a tuple of the format (roll, pitch)
        addr -- the 5 character address of the mimsy board being updated
        '''
        index = self.mapping.get(addr)
        if index is not None:
            proj, norm = Euler.fromAngles(data).rotate()
            self.get_vec(index).axis = proj
            self.setMimsyColor(index, b=0.5 + 0.5*norm)

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
            angle = Euler.fromAngles(rpy)
            self.angles.append(angle)
            self.set_vec(i, angle, initializing=True)

    def get_vec(self, i):
        '''
        Get the vector given by a linearized index.

        Keyword arguments:
        i -- the linearized index denoting the position of the vector
        '''
        return self.vecs[i]

    def set_vec(self, i, angle, initializing=False):
        '''
        Sets the vector given by a linearized index with the given
        Euler angles.

        Keyword arguments:
        i -- the linearized index denoting the position of the vector
        angle -- the Euler Angle object denoting the rotation of the vector
        '''
        x1, y1 = self.quadrant_coors(i)
        proj, norm = angle.rotate()
        v = arrow(pos=(x1,y1,Network.Z_OFFSET), axis=proj, shaftwidth=0.05, \
            color=(0, 0.5*(1 + norm), 0))
        if not initializing:
            self.setMimsyColor(i, b=0.5 + 0.5*norm)
        else:
            self.setMimsyColor(i, r=1)
        self.vecs.append(v)

    def setMimsyColor(self, i, r=0, g=0, b=0):
        '''
        Sets the color of a block given the linearized index corresponding
        with a specific mimsy board

        Keyword arguments:
        i -- the linearized index of the mimsy board / UI box
        r, g, b -- the red, green, and blue channels, respectively
        '''
        self.mimsies[i].color = (r, g, b)

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
        if self.inputs.shape != (self.w*self.h, Network.DIMENSION):
            raise ValueError('Bad Input: Input shape is ' + str(self.inputs.shape) \
                + ', should be (' + str(self.w*self.h) + ', ' + str(Network.DIMENSION)+ ')')


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

    @staticmethod
    def visualizeCSV(csv_path, out='output', extension='mp4'):
        '''
        Parses a text file for the network format and initializes a new network.


        Keyword arguments:
        csv -- a .CSV file with time-series data of mattress deformation data
        out -- the .mov file to output the animation to
        '''

        plt.xlim(-1,1)
        plt.ylim(-1,1)
        vec = plt.arrow(0, 0, 0, 0)

        i = 0
        for line in csv.reader(open(csv_path)):
            # mapping = dict(zip(line[::2], line[1::2]))
            time, x_a, y_a, z_a, temp, mid = line[1::2]
            time, x_a, y_a, z_a, temp = float(time), float(x_a), float(y_a), float(z_a), float(temp)
            roll, pitch = degrees(atan(y_a / z_a)), degrees(atan(-x_a / sqrt(y_a**2 + z_a**2)))
            angle = Euler(roll, pitch)
            proj, norm = angle.rotate()
            vec.remove()
            vec = plt.arrow(0, 0, *proj)
            plt.savefig('frame' + str(i) + '.png', facecolor=(1.0, 0.5+0.5*norm, 1.0), bbox_inches='tight')
            i += 1

        '''files = []
        for i, line in enumerate(csv): # FIXME: pseudocode
            plt.cla()
            plt.imshow(np.random.rand(5, 5), interpolation='nearest')
            fname = '_tmp%03d.png' % i
            print('Saving frame', fname)
            plt.savefig(fname)
            files.append(fname)
        anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                           interval=50, blit=False)'''

    ''' Class Methods '''

    @classmethod
    def initialize(cls, filename, testing):
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
        print(list(reversed(content)))
        for i, line in enumerate(list(reversed(content))):
            addrs = line.replace(" ", "").replace("\n", "").split(',')
            if not set:
                w = len(addrs)
                set = True
            elif w != len(addrs):
                raise RuntimeError('Setup file badly formatted: ' + \
                    'Inconsistent widths on lines 1 and ' + str(i))
            for j, addr in enumerate(addrs):
                mapping.put(addr, i*len(addrs)+j)
                if not re.match('[a-z0-9]{4}', addr):
                    raise RuntimeError('Setup file badly formatted: ' + \
                        'Mimsy board ID at position (' + \
                         str(j) + ', ' + str(i) + ') does not pass regex test.')
        return cls(w, h, mapping, testing)

if __name__ == '__main__':
    Network.visualizeCSV('test.csv')
