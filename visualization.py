import matplotlib.pyplot as plt
import csv, os, sys, imageio

imageio.plugins.ffmpeg.download()

import numpy as np
from math import sqrt, sin, cos, tan, atan, radians, degrees, pi

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
    '''A Stripped Network class.'''

    @staticmethod
    def visualizeCSV(csv_path, extension='mp4'):
        '''
        Parses a text file for the network format and initializes a new network.


        Keyword arguments:
        csv -- a .CSV file with time-series data of mattress deformation data
        out -- the .mov file to output the animation to
        '''

        plt.xlim(-1,1)
        plt.ylim(-1,1)
        vec = plt.arrow(0, 0, 0, 0)

        i, files = 0, []
        for line in csv.reader(open(csv_path)):
            # mapping = dict(zip(line[::2], line[1::2]))
            time, x_a, y_a, z_a, temp, mid = line[1::2]
            time, x_a, y_a, z_a, temp = float(time), float(x_a), float(y_a), float(z_a), float(temp)
            roll, pitch = degrees(atan(y_a / z_a)), degrees(atan(-x_a / sqrt(y_a**2 + z_a**2)))
            angle = Euler(roll, pitch)
            proj, norm = angle.rotate()
            vec.remove()
            vec = plt.arrow(0, 0, *proj)
            files.append('_frame' + str(i) + '.png') # TODO: tweak so fps converts to correct timestamp in seconds
            plt.savefig(files[i], facecolor=(1.0, 0.5+0.5*norm, 1.0), bbox_inches='tight')
            i += 1

        kargs = { 'macro_block_size' : None }
        writer = imageio.get_writer(csv_path.replace(".csv", "") + '.mp4', fps=1, **kargs)

        for im in reversed(files):
            print(im)
            writer.append_data(imageio.imread(im)[:, :, :])
        writer.close()

#        for file in files:
#            os.remove(file) # cleanup

if __name__ == '__main__':
    if len(sys.argv) == 1:
        Network.visualizeCSV('mimsy.csv')
    elif len(sys.argv) == 2:
        Network.visualizeCSV(sys.argv[1])
