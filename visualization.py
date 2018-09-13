import matplotlib.pyplot as plt
import csv, os, sys, warnings, traceback, imageio

imageio.plugins.ffmpeg.download()

import numpy as np
from math import sqrt, sin, cos, tan, atan, radians, degrees, pi
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

class Network:
    '''A Stripped Network class.'''

    @staticmethod
    def parseCSV(path):
        lowT, highT = float("inf"), float("-inf")
        csv_data = []
        for line in csv.reader(open(path)):
            time, x_a, y_a, z_a, temp, mID = line[1::2]
            time, x_a, y_a, z_a, temp = float(time), float(x_a), float(y_a), float(z_a), float(temp)
            if temp < lowT:
                lowT = temp - 1
            if temp > highT:
                highT = temp + 1
            csv_data.append((time, x_a, y_a, z_a, temp, mID))
        return csv_data, (lowT, highT)

    @staticmethod
    def temperatureGradientRGB(temp, min=0, max=200):
        assert (max > min and temp >= min and temp <= max)
        normalized = (temp - min) / float(max - min)
        return (1, 1-normalized, 1-normalized)

    @staticmethod
    def visualizeCSV(csv_path, output_path):
        '''
        Parses a text file for the network format and initializes a new network.


        Keyword arguments:
        csv -- a .CSV file with time-series data of mattress deformation data
        out -- the .mov file to output the animation to
        '''

        csv_data, temp_range = None, None
        try:
            csv_data, temp_range = Network.parseCSV(csv_path)
        except IOError:
            print("[Visualizer]: invalid CSV file.")
            traceback.print_exc()
            return

        plt.xlim(-1,1)
        plt.ylim(-1,1)
        vec = plt.arrow(0, 0, 0, 0)

        i, files, removed = 0, [], []
        # kargs = { 'macro_block_size' : None } # include **kargs in writer if running in linux
        writer = imageio.get_writer(os.path.join(output_path, os.path.basename(csv_path).replace(".csv", "") + '.mp4'), fps=1)
        for elem in csv_data:
            try:
                # mapping = dict(zip(line[::2], line[1::2]))
                time, x_a, y_a, z_a, temp, mid = elem
                try:
                    roll, pitch = degrees(atan(y_a / z_a)), degrees(atan(-x_a / sqrt(y_a**2 + z_a**2)))
                except:
                    print("[Visualization]: Line " + str(i) + " at time " + str(time) + " had invalid roll/pitch projection, skipping.")
                    continue
                angle = Euler(roll, pitch)
                proj, norm = angle.rotate()
                vec.remove()
                vec = plt.arrow(0, 0, *proj, head_width=0.03, head_length=0.05)
                rgb = Network.temperatureGradientRGB(temp, *temp_range)
                plt.title("Time (s): " + str(time) + " | Temp ($^\circ$C): " + str(temp))
                fname = '_frame' + str(i) + '.png'
                files.append(fname) # TODO: tweak so fps converts to correct timestamp in seconds
                plt.savefig(files[i], figsize=(6.66, 4.35), dpi=100, facecolor=rgb, bbox_inches='tight')
                writer.append_data(imageio.imread(files[i])[:, :, :])
                os.remove(files[i])
                removed.append(fname)
                i += 1
            except:
                print("[Visualizer]: something went wrong.")
                traceback.print_exc()
        writer.close()

        if len(files) != len(removed): # error-handling cleanup
            for file in files:
                if not removed.contains(file):
                    os.remove(file)

if __name__ == '__main__':
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    Tk().withdraw()
    csv_path = askopenfilename(initialdir="C:/Users/%s" % getuser(), title="Select your data file to visualize.", filetypes=(("CSV Files", "*.csv"),))
    output_path = askdirectory(initialdir=csv_path, title="Select where the visualization should be saved.")
    Network.visualizeCSV(csv_path, output_path)
