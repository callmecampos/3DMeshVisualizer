import serial, threading, struct, os, sys, warnings, traceback
import datetime, binascii, random, re, keyboard
import numpy as np
from visual import *
from math import sqrt, sin, cos, tan, atan, radians, sqrt, pi
from time import time, sleep
from bidict import bidict
from Tkinter import Tk
from tkFileDialog import askopenfilename

''' UTILS '''

init_angles, reset_flag, reset_buf, step = {}, True, [], False

def argmax(iterable, f=lambda x : x):
    return max(enumerate(map(f, iterable)), key=lambda x: x[1])[0]

def areAll(iterable, val=True):
    for elem in iterable:
        if elem != val:
            return False
    return True

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
        return (result[1] * Euler.SCALE, -result[0] * Euler.SCALE, 0), \
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

        self.gui = False

    def __len__(self):
        return self.w * self.h

    def setupGUI(self):
        '''
        Visual Python GUI setup.
        '''
        self.gui = True
        
        self.scene = display(title='Network' + str(Network._id) + 'Visualization', x=0, y=0)
        self.scene.background = (0.5,0.5,0.5)

        cx, cy = float(self.w) / 2 - .5, float(self.h) / 2 - .5
        cz = self.scene.center.z
        self.center = (cx, cy, cz)
        self.scene.center = (cx,cy,cz)
        self.scene.forward = (0,0,-1)
        self.scene.range = max(self.w, self.h)

        self.scene.lights = [vector(1,0,0), vector(0, 1, 0), vector(0, 0, 1), \
            vector(-1,0,0), vector(0, -1, 0), vector(0, 0, -1)]
        self.scene.ambient = 0

        self.mimsies = []
        for i in range(self.w*self.h):
            x0, y0 = self.quadrant_coors(i)
            b = box(pos=(x0,y0,0), length=1, height=1, width=0.2, color=(1,0,0))
            self.mimsies.append(b)
            # print("We did it {}".format(w*h))

        self.inputs = np.array([(0, 0) for k in range(self.w*self.h)])
        self.initGUI()

        Network._id += 1

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
            # print('set color.')

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

    ''' Class Methods '''

    @classmethod
    def initialize(cls):
        '''
        Parses a text file for the network format and initializes a new network.
        '''
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        Tk().withdraw()
        setup_path = askopenfilename(initialdir="./", title="Select your setup.txt file.", filetypes=(("Text files", "*.txt"),))

        f = open(setup_path, "r")
        content = f.readlines()

        h = len(content)
        w = 0
        set = False

        mapping = bidict({})
        # print(list(reversed(content)))
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
        return cls(w, h, mapping, False)

class OpenHdlc(object):

    HDLC_FLAG              = '\x7e'
    HDLC_FLAG_ESCAPED      = '\x5e'
    HDLC_ESCAPE            = '\x7d'
    HDLC_ESCAPE_ESCAPED    = '\x5d'
    HDLC_CRCINIT           = 0xffff
    HDLC_CRCGOOD           = 0xf0b8

    FCS16TAB  = (
        0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,
        0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
        0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,
        0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
        0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,
        0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
        0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,
        0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
        0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,
        0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
        0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,
        0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
        0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,
        0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
        0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,
        0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
        0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,
        0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
        0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,
        0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
        0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,
        0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
        0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,
        0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
        0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,
        0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
        0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,
        0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
        0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,
        0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
        0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,
        0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78,
    )

    #============================ public ======================================

    def hdlcify(self,inBuf):
        '''
        Build an hdlc frame.

        Use 0x00 for both addr byte, and control byte.
        '''

        # make copy of input
        outBuf     = inBuf[:]

        # calculate CRC
        crc        = self.HDLC_CRCINIT
        for b in outBuf:
            crc    = self._crcIteration(crc,b)
        crc        = 0xffff-crc

        # append CRC
        outBuf     = outBuf + chr(crc & 0xff) + chr((crc & 0xff00) >> 8)

        # stuff bytes
        outBuf     = outBuf.replace(self.HDLC_ESCAPE, self.HDLC_ESCAPE+self.HDLC_ESCAPE_ESCAPED)
        outBuf     = outBuf.replace(self.HDLC_FLAG,   self.HDLC_ESCAPE+self.HDLC_FLAG_ESCAPED)

        # add flags
        outBuf     = self.HDLC_FLAG + outBuf + self.HDLC_FLAG

        return outBuf

    def dehdlcify(self,inBuf):
        '''
        Parse an hdlc frame.

        :returns: the extracted frame, or -1 if wrong checksum
        '''
        assert inBuf[ 0]==self.HDLC_FLAG
        assert inBuf[-1]==self.HDLC_FLAG

        # make copy of input
        outBuf     = inBuf[:]

        # remove flags
        outBuf     = outBuf[1:-1]

        # unstuff
        outBuf     = outBuf.replace(self.HDLC_ESCAPE+self.HDLC_FLAG_ESCAPED,   self.HDLC_FLAG)
        outBuf     = outBuf.replace(self.HDLC_ESCAPE+self.HDLC_ESCAPE_ESCAPED, self.HDLC_ESCAPE)

        if len(outBuf)<2:
            raise Exception('packet too short')

        # check CRC
        crc        = self.HDLC_CRCINIT
        for b in outBuf:
            crc    = self._crcIteration(crc,b)
        if crc!=self.HDLC_CRCGOOD:
           raise Exception('wrong CRC')

        # remove CRC
        outBuf     = outBuf[:-2] # remove CRC

        return [ord(b) for b in outBuf]

    #============================ private =====================================

    def _crcIteration(self,crc,b):
        return (crc>>8)^self.FCS16TAB[((crc^(ord(b))) & 0xff)]

class moteProbe(threading.Thread):

    CMD_SET_DAGROOT = '7e5259bbbb00000000000001deadbeefcafedeadbeefcafedeadbeefa7d97e' # prefix: bbbb000000000000 keyindex : 01 keyvalue: deadbeefcafedeadbeefcafedeadbeef
    CMD_SEND_DATA   = '7e44141592000012e63b78001180bbbb0000000000000000000000000001bbbb000000000000141592000012e63b07d007d0000ea30d706f69706f697a837e'
    SLOT_DURATION   = 0.015
    UINJECT_MASK    = 'uinject'

    init_angles, calibrating, reset_flag, reset_buf = {}, True, True, []
    x_c, y_c, z_c, i_c, state = 0, 0, 0, 0, 0
    t_init = time()
    g_index, forward_index, side_index = 2, 0, 1
    g_sign, f_sign, s_sign = 1, 1, 1

    def __init__(self,serialport=None, network=None):

        # store params
        self.serialport           = serialport
        self.network              = network
        self.calibrating          = True # FIXME: check for preset text file
        self.calib_rotation       = { elem : np.eye(3) for elem in network.mapping.keys() }

        self.x_c = { elem : 0 for elem in network.mapping.keys() }
        self.y_c = { elem : 0 for elem in network.mapping.keys() }
        self.z_c = { elem : 0 for elem in network.mapping.keys() }
        self.i_c = { elem : 0 for elem in network.mapping.keys() }

        # TODO: see if hash is present in files with .out type, ask if they want to use a present preset
        preset = self.checkForPreset()
        if preset:
            print("Calibration preset discovered for this setup file.")
            use = raw_input("Do you want to use the calibration parameters provided by this previous calibration?[y/n] ")
            if use == 'y':
                print("Loading calibration preset parameters.")
                self.calibrating = False
                self.calib_rotation = dict(preset[()])
            else:
                print("Ok, starting calibration.")

        print(self.calib_rotation)

        self.network.setupGUI()

        self.data                 = []
        self.now                  = datetime.datetime.now()

        # local variables
        self.hdlc                 = OpenHdlc()
        self.lastRxByte           = self.hdlc.HDLC_FLAG
        self.busyReceiving        = False
        self.inputBuf             = ''
        self.last_counter         = None
        self.outputBuf            = [binascii.unhexlify(self.CMD_SET_DAGROOT)]
        self.outputBufLock        = threading.RLock()
        self.dataLock             = threading.Lock()

        # flag to permit exit from read loop
        self.goOn                 = True

        # initialize the parent class
        threading.Thread.__init__(self)

        # give this thread a name
        self.name                 = 'moteProbe@'+self.serialport
        #print "counter latency(second)"
        # start myself
        self.start()

    #======================== thread ==========================================

    def run(self):
        now = self.now
        myfilename = str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"-"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)+".csv"
        self.myfile = open(myfilename,"w")

        try:

            while self.goOn:     # open serial port

                self.serial = serial.Serial(self.serialport,'115200')
                print "Network Forming..."
                while self.goOn: # read bytes from serial port

                    try:
                        rxByte = self.serial.read(1)
                    except Exception as err:
                        print err
                        sleep(1)
                        break
                    else:
                        if      (
                                    (not self.busyReceiving)             and
                                    self.lastRxByte==self.hdlc.HDLC_FLAG and
                                    rxByte!=self.hdlc.HDLC_FLAG
                                ):
                            # start of frame
                            self.busyReceiving       = True
                            self.inputBuf            = self.hdlc.HDLC_FLAG
                            self.inputBuf           += rxByte
                        elif    (
                                    self.busyReceiving                   and
                                    rxByte!=self.hdlc.HDLC_FLAG
                                ):
                            # middle of frame

                            self.inputBuf           += rxByte
                        elif    (
                                    self.busyReceiving                   and
                                    rxByte==self.hdlc.HDLC_FLAG
                                ):
                            # end of frame
                            self.busyReceiving       = False
                            self.inputBuf           += rxByte

                            try:
                                tempBuf              = self.inputBuf
                                self.inputBuf        = self.hdlc.dehdlcify(self.inputBuf)
                            except Exception as err:
                                pass
                                # print '{0}: invalid serial frame: {2} {1}'.format(self.name, err, tempBuf)
                            else:
                                if   self.inputBuf==[ord('R')]:
                                    with self.outputBufLock:
                                        if self.outputBuf:
                                            outputToWrite = self.outputBuf.pop(0)
                                            self.serial.write(outputToWrite)
                                elif self.inputBuf[0]==ord('D'):
                                    if self.UINJECT_MASK == ''.join(chr(i) for i in self.inputBuf[-7:]):
                                        asn_initial  = struct.unpack('<HHB',''.join([chr(c) for c in self.inputBuf[3:8]]))

                                        Xaccel  = struct.unpack('<h',''.join([chr(b) for b in self.inputBuf[-9:-7]]))[0]
                                        Xaccel = Xaccel / 16000.0
                                        Yaccel  = struct.unpack('<h',''.join([chr(b) for b in self.inputBuf[-11:-9]]))[0]
                                        Yaccel = Yaccel / 16000.0
                                        Zaccel  = struct.unpack('<h',''.join([chr(b) for b in self.inputBuf[-13:-11]]))[0]
                                        Zaccel = Zaccel / 16000.0

                                        global step

                                        # first, ask to leave mattress flat for 5 seconds, press any key when done
                                        # second, tilt mattress forward 90 degrees for 5 seconds, press key when done
                                        # third, tilt mattress left/right? 90 degrees for 5 seconds, press key when done
                                        # then, save calibration params and set calibrating false
                                        if self.calibrating:
                                            address = str('{:x}'.format(struct.unpack('<H',''.join([chr(b) for b in self.inputBuf[-15:-13]]))[0]))
                                            if self.state <= 1: # leave mattress flat
                                                if self.state == 0:
                                                    step = False
                                                    print('Welcome to the calibration process.')
                                                    print('Please leave the mattress flat until the first stage is complete.')
                                                    print('Press the "s" key when ready to begin the first stage.')
                                                    self.state = 1
                                                elif step:
                                                    self.x_c[address] += Xaccel
                                                    self.y_c[address] += Yaccel
                                                    self.z_c[address] += Zaccel
                                                    self.i_c[address] += 1
                                                
                                                if areAll([elem > 6 for elem in self.i_c.values()]):
                                                    for addr in self.i_c.keys():
                                                        self.x_c[addr] = self.x_c[addr] / self.i_c[addr]
                                                        self.y_c[addr] = self.y_c[addr] / self.i_c[addr]
                                                        self.z_c[addr] = self.z_c[addr] / self.i_c[addr]
                                                        a_vec = [self.x_c[addr], self.y_c[addr], self.z_c[addr]]
                                                        self.calib_rotation[addr][:,2] = np.array(a_vec)
                                                        self.x_c[addr], self.y_c[addr], self.z_c[addr], self.i_c[addr] = 0, 0, 0, 0
                                                    print('Done with stage 1.')
                                                    self.state = 2
                                            elif self.state <= 3: # tilt mattress forward
                                                if self.state == 2:
                                                    step = False
                                                    print('Please tilt the mattress forwards 90 degrees, maintaining its position until the second stage is complete.')
                                                    print('Press the "s" key when ready to begin the second stage.')
                                                    self.state = 3
                                                elif step:
                                                    self.x_c[address] += Xaccel
                                                    self.y_c[address] += Yaccel
                                                    self.z_c[address] += Zaccel
                                                    self.i_c[address] += 1
                                                    
                                                if areAll([elem > 6 for elem in self.i_c.values()]):
                                                    for addr in self.i_c.keys():
                                                        self.x_c[addr] = self.x_c[addr] / self.i_c[addr]
                                                        self.y_c[addr] = self.y_c[addr] / self.i_c[addr]
                                                        self.z_c[addr] = self.z_c[addr] / self.i_c[addr]
                                                        a_vec = [self.x_c[addr], self.y_c[addr], self.z_c[addr]]
                                                        self.calib_rotation[addr][:,1] = np.array(a_vec)
                                                        self.x_c[addr], self.y_c[addr], self.z_c[addr], self.i_c[addr] = 0, 0, 0, 0
                                                    print('Done with stage 2.')
                                                    self.state = 4
                                            elif self.state <= 5: # tilt mattress left
                                                if self.state == 4:
                                                    step = False
                                                    print('Please tilt the mattress right 90 degrees, maintaining its position until the second stage is complete.')
                                                    print('Press the "s" when ready to begin the final stage of calibration.')
                                                    self.state = 5
                                                elif step:
                                                    self.x_c[address] += Xaccel
                                                    self.y_c[address] += Yaccel
                                                    self.z_c[address] += Zaccel
                                                    self.i_c[address] += 1
                                                    
                                                if areAll([elem > 6 for elem in self.i_c.values()]):
                                                    for addr in self.i_c.keys():
                                                        self.x_c[addr] = self.x_c[addr] / self.i_c[addr]
                                                        self.y_c[addr] = self.y_c[addr] / self.i_c[addr]
                                                        self.z_c[addr] = self.z_c[addr] / self.i_c[addr]
                                                        a_vec = [self.x_c[addr], self.y_c[addr], self.z_c[addr]]
                                                        self.calib_rotation[addr][:,0] = np.array(a_vec)
                                                        self.x_c[addr], self.y_c[addr], self.z_c[addr], self.i_c[addr] = 0, 0, 0, 0
                                                    print('Done with stage 3.')
                                                    self.state = 6
                                                    self.calibrating = False

                                                    self.saveRotation()
                                            else:
                                                self.calibrating = False
                                        else:
                                            if not self.network.gui:
                                                self.network.setupGUI()
                                            
                                            formattedAddr = str('{:x}'.format(struct.unpack('<H',''.join([chr(b) for b in self.inputBuf[-15:-13]]))[0]))

                                            # print("init: {} {} {}".format(Xaccel, Yaccel, Zaccel))
                                            
                                            vecAccel = np.array([Xaccel, Yaccel, Zaccel])
                                            Xaccel, Yaccel, Zaccel = np.dot(np.linalg.pinv(self.calib_rotation[formattedAddr]), vecAccel)

                                            # print("rotated: {} {} {}".format(Xaccel, Yaccel, Zaccel)) # DEBUG
                                            
                                            roll, pitch = atan(Yaccel/Zaccel)*180.0/3.14, \
                                                        atan(-Xaccel/sqrt(Yaccel**2 + Zaccel**2))*180.0/3.14
                                            temperature = struct.unpack('<h',''.join([chr(b) for b in self.inputBuf[-17:-15]]))[0]

                                            global reset_flag
                                            
                                            if reset_flag: # start calibrating again
                                                reset_flag = False
                                                if raw_input("Are you sure you want to recalibrate? [y/n]") == "y":
                                                    self.state = 0
                                                    self.calibrating = True
                                                    step = False

                                                    self.x_c = { elem : 0 for elem in network.mapping.keys() }
                                                    self.y_c = { elem : 0 for elem in network.mapping.keys() }
                                                    self.z_c = { elem : 0 for elem in network.mapping.keys() }
                                                    self.i_c = { elem : 0 for elem in network.mapping.keys() }
                                                else:
                                                    print("Ok, not calibrating.")

                                            data = "Time[s]," + str((asn_initial[0] + asn_initial[1]*65536)*0.01) + \
                                                    ",Xacceleration[gs]," + str(Xaccel) + ",Yacceleration[gs]," + str(Yaccel) + \
                                                    ",Zacceleration[gs]," + str(Zaccel) + ",roll[deg]," + str(roll) + \
                                                    ",pitch[deg]," + str(roll) + \
                                                    ",Temperature[C]," + str(temperature) + \
                                                    ",Address," + formattedAddr

                                            if self.last_counter!=None:
                                                if counter-self.last_counter!=1:
                                                    pass
                                            if True:
                                                print data
                                                print formattedAddr, roll, pitch

                                            self.myfile.write(data + "\n")
                                            with self.outputBufLock:
                                                self.outputBuf += [binascii.unhexlify(self.CMD_SEND_DATA)]
                                            self.network.update(data=(roll, pitch), addr=formattedAddr)


                        self.lastRxByte = rxByte

        except Exception as err:
            traceback.print_exc()

    #======================== public ==========================================

    def close(self):
        self.goOn = False

    def saveRotation(self):
        '''
        Write mimsy list hash mapping to rotation matrix to file.
        '''
        mimsys = self.network.mapping.keys()
        dim = (self.network.w, self.network.h)
        rotation = self.calib_rotation
        
        h = hash(tuple([dim[0], dim[1]] + mimsys))
        np.save('{}.npy'.format(h), rotation)

    def checkForPreset(self):
        '''
        Check if current mimsy organization has been previously calibrated.
        '''
        mimsys = self.network.mapping.keys()
        dim = (self.network.w, self.network.h)
        
        # list of files of .out type
        h = hash(tuple([dim[0], dim[1]] + mimsys))
        for filename in os.listdir("."):
            if filename.endswith(".npy"):
                if os.path.splitext(filename)[0] == str(h):
                    return np.load("{}.npy".format(os.path.splitext(filename)[0]))

    #======================== private =========================================

def main():
    print 'poipoi'

def reset(event):
    global reset_flag
    global reset_buf
    if event.event_type == 'down':
        reset_flag = True
        reset_buf = []

def calibrationStep(event):
    global step
    if event.event_type == 'down':
        step = True
        print('S pressed, recording orientation data.')

if __name__ == "__main__":
    network = Network.initialize()
    init_angles, reset_flag, reset_buf = {}, False, []
    
    keyboard.hook_key('r', lambda x: reset(x), suppress=False)
    keyboard.hook_key('s', lambda x: calibrationStep(x), suppress=False)
    if False:
        print('DAG root not detected, please check your configuration.')
    else:
        moteProbe("COM4", network)
