from visual import *
import numpy as np
import random, re, csv, os, sys, keyboard, warnings, traceback
from math import sqrt, sin, cos, tan, atan, radians, pi
from bidict import bidict
from getpass import getuser
from Tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

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

class Mimsy:
    ''' A Mimsy class '''

    def __init__(self, addr, g_x, g_y, g_z, roll, pitch, yaw, time, roll_off, pitch_off, previous=None):
        self.addr = addr
        self.t = time
        self.euler = (roll, pitch, yaw)
        self.euler_offset = (roll_off, pitch_off)
        self.accelerometer = (g_x, g_y, g_z)
        self.previous = previous

    def roll(self):
        return self.euler[0] - self.euler_offset[0]

    def pitch(self):
        return self.euler[1] - self.euler_offset[1]

    def euler(self):
        return (self.roll, self.pitch)

    def time(self):
        return self.t

    def address(self):
        return self.addr

    def previous(self):
        return self.prev

    def setPrevious(self, prev):
        self.prev = prev

    def hasPrevious(self):
        return self.previous() is not None

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
    def parseCSV(path):
        extract_float = lambda string: float(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)[0])
        csv_data = []
        for line in csv.reader(open(path)):
            addr = line[7]
            g_x, g_y, g_z, roll, pitch, yaw, time, roll_off, pitch_off = (extract_float(elem) for i, elem in enumerate(line) if i != 7)
            csv_data.append(Mimsy(addr, g_x, g_y, g_z, roll, pitch, yaw, time, roll_off, pitch_off))
        return csv_data

    ''' Class Methods '''

    @classmethod
    def initialize(cls, filename, testing=False):
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

def peer(dataCSV, writeCSV, setup):
    '''
    Parses a past CSV file with mimsy data, let's user click
    back and forth through data. Key-action-mappings below:

    'a'  -- backward 10 seconds
    'd'  -- forward 10 seconds
    'w' -- backward 1 mimsy update
    's' -- forward 1 mimsy update
    ' ' -- save current mimsy state

    Keyword arguments:
    dataCSV -- the relative path of the .csv file filled with mimsy data to parse
    writeCSV -- the relative path of a new (or existing) .csv file to write corresponding mimsy data to
    setup -- the relative path of the setup file with mimsy-mattress location data
    '''

    network = Network.initialize(setup)
    state_map = {address: None for address in network.mapping.keys()}

    csv_data = None
    try:
        csv_data = Network.parseCSV(dataCSV)
    except IOError:
        print("[Visualizer]: invalid CSV file.")
        traceback.print_exc()
        return

    update_latency_threshold = 60

    def forward_update(i):
        mimsy = csv_data[i]
        network.update(data=mimsy.euler(), addr=mimsy.address())
        if not mimsy.hasPrevious():
            mimsy.setPrevious(state_map[mimsy.address()])
        state_map[mimsy.address()] = mimsy
        return mimsy

    def backward_update(i):
        mimsy = csv_data[i]
        if mimsy.hasPrevious():
            network.update(data=mimsy.previous().euler(), addr=mimsy.previous.address())
            state_map[mimsy.address()] = mimsy.previous()
        else:
            network.update(data=(0.0, 0.0), addr=mimsy.address())
            network.setMimsyColor(network.mapping.get(mimsy.address()), r=1) # hasn't joined network yet
        return mimsy.previous()

    # open csv
    with open(writeCSV, 'w+') as out:
        i, time = -1, -1
        while i < len(csv_data):
            key = keyboard.read_key() # wait for key press
            if key == 'a': # move backward ten seconds
                if i <= 0:
                    print("You're already at the beginning.")
                    continue
                new_time = time
                while i > 0 and time - new_time < 10.0:
                    previous = backward_update(i)
                    new_time = max([m.time() for m in state_map.values() if m])
                    i -= 1
                    for m in state_map.values():
                        if m and new_time - m.time() >= update_latency_threshold:
                            network.setMimsyColor(network.mapping.get(m.address()), r=1) # fell out of network
                time = new_time
            elif key == 'd': # move forward ten seconds
                new_time = time
                while i < len(csv_data) and new_time - time < 10.0:
                    i += 1; new_time = forward_update(i).time()
                    for m in state_map.values():
                        if m and new_time - m.time() >= update_latency_threshold:
                            network.setMimsyColor(network.mapping.get(m.address()), r=1) # fell out of network
                time = new_time
            elif key == 'j': # move backward one mimsy update
                if i == 0:
                    print("You're already at the beginning.")
                    continue
                previous = backward_update(i)
                time = max([m.time() for m in state_map.values() if m])
                i -= 1
                for m in state_map.values():
                    if m and time - m.time() >= update_latency_threshold:
                        network.setMimsyColor(network.mapping.get(m.address()), r=1) # fell out of network
            elif key == 'l': # move forward one mimsy update
                i += 1; time = update_network(i).time()
                for m in state_map.values():
                    if m and time - m.time() >= update_latency_threshold:
                        network.setMimsyColor(network.mapping.get(m.address()), r=1) # fell out of network
            elif key == 'p': # write current state to csv
                out.write(str(state_map) + '\n')
                print('Wrote to csv at ASN: ' + str(time))

if __name__ == '__main__':
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    Tk().withdraw()
    csv_path = askopenfilename(initialdir="C:/Users/%s" % getuser(), title="Select your data file to visualize.", filetypes=(("CSV Files", "*.csv"),))
    output_path = askdirectory(initialdir=csv_path, title="Select where the output should be saved.")
    peer(csv_path, output_path)
