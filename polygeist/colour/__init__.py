"""
    Polygeist Colour Transforms Library
    Copyright 2023 Polygeist LTD
    Author: blitmaps

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

"""
import os.path as path
import unittest

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import Planck, speed_of_light, Boltzmann

# Get the path to the library root, so we can load files
path_to_library_dir = path.dirname(path.abspath(__file__))

# Range of human vision, in nanometres
visible_range_nm = [390, 730]
# Visible range over which to compute illuminant functions
visible_range = np.arange(visible_range_nm[0], visible_range_nm[1])


class Illuminant:
    """
        Contains standard illumination functions, and the means to compute them.
    """

    def __init__(self):
        self._D65 = np.genfromtxt(f'{path_to_library_dir}/CIE_std_illum_D65.csv', delimiter=",")
        pass

    def get(self, function_name):
        """
        Get a standard illumination, specified by name (e.g. "D65")
        @arg function_name: Name of illumination (e.g. "D65") (string)
        @return: Illumination (np.array)
        """
        if "D65" in function_name:
            fD65 = interp1d(self._D65[:, 0], self._D65[:, 1], kind='cubic')
            return fD65(visible_range)
        if "B5000K" in function_name:
            return np.array(self.planck(visible_range / 1e9, 5000.))
        return None

    @staticmethod
    def planck(wavelengths, temperature):
        """
        Get a blackbody radiator at a given temperature
        @arg wavelengths: wavelengths (np.array)
        @arg temperature: temperature of the radiator (np.array)
        @return: radiation (np.array)
        """
        h = Planck  # Js
        c = speed_of_light  # m/s
        kb = Boltzmann  # J/molec K

        return 8. * np.pi * h * c / (wavelengths ** 5. * (np.exp(h * c / (wavelengths * kb * temperature)) - 1.))

    @staticmethod
    def from_file_with_range(filename, wavelength_range, interp='cubic'):
        """
        Create spectra from csv file with given spectral range
        file e.g.::
            390, 0.1
            500, 0.5

        @arg filename: path (string)
        @arg wavelength_range: wavelengths e.g. np.arange(380, 781) (np.array)
        @arg interp: way of smoothing spectra (see interp1d) (string)
        @return: radiation/absorption over wavelengths (np.array)
        """
        file = np.genfromtxt(filename, delimiter=",")
        unique_x = file[np.unique(file[:, 0], return_index=True)[1], :]
        model = interp1d(unique_x[:, 0], unique_x[:, 1], kind=interp, fill_value=np.nan, bounds_error=False)
        return model(wavelength_range)

    @staticmethod
    def merge_spectra(spectra_a, spectra_b):
        a = spectra_a.copy()
        b = spectra_b.copy()
        a[np.isnan(a)] = 0
        b[np.isnan(b)] = 0
        a[a < 0] = 0
        b[b < 0] = 0
        return a + b


class CMFs:
    """
        Contains the standard colour matching functions (for humans).
    """

    def __init__(self):
        self._CIE_1931_2Deg = np.genfromtxt(f'{path_to_library_dir}/CIE1931.csv', delimiter=",")
        '''Load 1931 2deg Observer functions'''

    def get(self, function_name):
        """
        Get a set of colour matching functions, specified by name (e.g. "CIE_1931_2deg")
        @arg function_name: Name of CMFs (e.g. "CIE_1931_2deg") (string)
        @return: list of functions (e.g: [X, Y, Z]) (np.array, np.array, np.array)
        """
        if "CIE_1931_2deg" in function_name:
            fX = interp1d(self._CIE_1931_2Deg[:, 0], self._CIE_1931_2Deg[:, 1], kind='cubic')
            fY = interp1d(self._CIE_1931_2Deg[:, 0], self._CIE_1931_2Deg[:, 2], kind='cubic')
            fZ = interp1d(self._CIE_1931_2Deg[:, 0], self._CIE_1931_2Deg[:, 3], kind='cubic')
            X = fX(visible_range)
            Y = fY(visible_range)
            Z = fZ(visible_range)
            return [X, Y, Z]


class Convert:
    @staticmethod
    def XYZ_to_Yxy(XYZ):
        """
        Convert between XYZ and Yxy (2d representation of tristimulus values)
        @arg XYZ: image of XYZ values (np.array IxJx3)
        @return: image of Yxy values (np.array IxJx3)
        """
        copy = XYZ.copy()
        copy[:, :, 0] = copy[:, :, 0] + copy[:, :, 1] + copy[:, :, 2]
        copy[:, :, 1] = XYZ[:, :, 0] / copy[:, :, 0]
        copy[:, :, 2] = XYZ[:, :, 1] / copy[:, :, 0]
        copy[:, :, 0] = XYZ[:, :, 1]
        return copy

    @staticmethod
    def XYZ_to_Linear(XYZ, transform):
        """
           Converts an image or array of XYZ values to an arbitrary colour space that is linear
           @arg XYZ: list or image of XYZ values (np.array Nx3 or IxJx3)
           @arg transform: matrix to go from XYZ -> Space (np.matrix 3x3)
           @return: list or image of Space values (np.array Nx3 or IxJx3)
        """
        if len(XYZ.shape) < 3:
            return np.matmul(transform, [XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]])
        else:
            img2 = XYZ.reshape((XYZ.shape[0] * XYZ.shape[1], 3))
            output = np.matmul(transform, [img2[:, 0], img2[:, 1], img2[:, 2]])
            return np.array(output.T).reshape(XYZ.shape)

    @staticmethod
    def XYZ_to_sRGB(XYZ):
        """
           Converts an image or array of XYZ values to linear sRGB (no compression / adaptation or gamma)
           @arg XYZ: list or image of XYZ values (np.array Nx3 or IxJx3)
           @return: list or image of sRGB values (np.array Nx3 or IxJx3)
        """
        transform = np.matrix([[3.2404542, -1.5371385, -0.4985314],
                               [-0.969266, 1.8760108, 0.041556],
                               [0.0556434, -0.2040259, 1.0572252]])
        return Convert.XYZ_to_Linear(XYZ, transform)

    @staticmethod
    def sRGB_to_XYZ(sRGB):
        """
           Converts an image or array of sRGB values to linear XYZ (no compression / adaptation or gamma)
           @arg sRGB: list or image of sRGB values (np.array Nx3 or IxJx3)
           @return: list or image of XYZ values (np.array Nx3 or IxJx3)
        """
        transform = np.linalg.inv(np.matrix([[3.2404542, -1.5371385, -0.4985314],
                                             [-0.969266, 1.8760108, 0.041556],
                                             [0.0556434, -0.2040259, 1.0572252]]))
        return Convert.XYZ_to_Linear(sRGB, transform)

    @staticmethod
    def Yxy_to_XYZ(Yxy):
        """
        Convert between XYZ and Yxy (2d representation of tristimulus values)
        @arg Yxy: list or image of Yxy values (np.array Nx3 or IxJx3)
        @return: list or image of Yxy values (np.array Nx3 or IxJx3)
        """
        if len(Yxy.shape) < 3:
            X = (Yxy[:, 1] * Yxy[:, 0]) / Yxy[:, 2]
            Y = Yxy[:, 0]
            Z = ((1. - Yxy[:, 1] - Yxy[:, 2]) * Yxy[:, 0]) / Yxy[:, 2]
            return X, Y, Z
        else:
            copy = Yxy.copy()
            copy[:, :, 0] = (Yxy[:, :, 1] * Yxy[:, :, 0]) / Yxy[:, :, 2]
            copy[:, :, 1] = Yxy[:, :, 0]
            copy[:, :, 2] = ((1. - Yxy[:, :, 1] - Yxy[:, :, 2]) * Yxy[:, :, 0]) / Yxy[:, :, 2]
            return copy

    @staticmethod
    def build_transform(tri_square_1, tri_square_2):
        """
        Build transform matrices between two spaces, each must be equal in N dimension.
        Can be photometric or colourimetric.
        e.g.::

            Identity matrix (Max RGB)              Specific Values
            A = [1, 0, 0;                        X = [Xr1, Xr2, Xr3;
                 0, 1, 0;                             Xg2, Xg2, Xg3;
                 0, 0, 1 ];                           Xb1, Xb2, Xb3;]

            Solves mA = X

            return m m^-1

        @arg tri_square_1: First space tristimulus array  (np.array Nx3)
        @arg tri_square_2: Second space tristimulus array  (np.array Nx3)
        @return: transform and inverse transform matrix (np.array Nx3)
        """
        m = np.linalg.solve(tri_square_1, tri_square_2)
        m_inv = np.linalg.inv(m)
        return m, m_inv

    @staticmethod
    def apply(tri, matrix):
        """
            More attractive syntax for transform
            @arg tri: Tristimulus values (np.array Nx3)
            @arg matrix: Matrix to transform (np.array 3x3)
            @return: transformed values (np.array Nx3)

        """
        return np.matmul(matrix, tri.T)


def integrate(spectra_a, spectra_b):
    """
    Integrates 2 spectra with the same scale, using the trapezium rule.
    @arg spectra_a: Array of illumination power by nm (np.array)
    @arg spectra_b: Array of sensor response by nm  (np.array)
    @return: Sensor response (float)
    """
    a = spectra_a.copy()
    b = spectra_b.copy()
    a[np.isnan(a)] = 0
    b[np.isnan(b)] = 0
    return np.trapz(np.multiply(a, b))


class ColourTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmfs = CMFs()
        self.illuminants = Illuminant()

    def test_d65_cie2(self):
        """
            Test illumination D65 against 1931 CMFs 2 deg
        """
        # Load CMFs
        X, Y, Z = self.cmfs.get("CIE_1931_2deg")
        # Load D65
        D65 = self.illuminants.get("D65")
        # Human observer observes D65
        obs = [integrate(X, D65), integrate(Y, D65), integrate(Z, D65)]
        # is little x correct (X / (X + Y + Z))
        self.assertLessEqual(np.abs((obs[0] / np.sum(obs)) - 0.3128), 0.001)
        # is little y correct (Y / (X + Y + Z))
        self.assertLessEqual(np.abs((obs[1] / np.sum(obs)) - 0.3290), 0.001)

    def test_XYZ_to_Yxy_list(self):
        """
            Test conversion between a list of XYZs and Yxy
        """
        Yxys = Convert.XYZ_to_Yxy(np.array([[50., 50., 50.], [60., 60., 60.]]))

        # Luminance should be the same
        self.assertEqual(Yxys[0][0], 50)
        self.assertEqual(Yxys[1][0], 60)

        # The result should be the centre of colour space (.33r for each coordinate)
        self.assertLessEqual(np.abs(Yxys[0][1] - .3333), 0.0001)

    def test_Yxy_to_XYZ_list(self):
        """
            Test conversion between a list of Yxys and XYZ
        """
        # EE White has xy .33 and results in X = Y = Z
        v3 = .3333333333
        # XYZ should be 50 50 50 and 60 60 60, one test is sufficient, as if incorrect both will be incorrect
        X, Y, Z = Convert.Yxy_to_XYZ(np.array([[50, v3, v3], [60, v3, v3]]))
        self.assertLessEqual(abs(X[0] - 50.), .01)
        self.assertLessEqual(abs(Z[0] - 50.), .01)

    def test_Yxy_to_XYZ_image(self):
        """
            Test conversion between an image of Yxys and XYZ
        """
        # As above, but for image dimensions
        v3 = .3333333333
        image = np.ones((1, 1, 3))
        image[:, :, 0] *= 50
        image[:, :, 1] *= v3
        image[:, :, 2] *= v3

        IXYZ = Convert.Yxy_to_XYZ(image)

        self.assertLessEqual(abs(IXYZ[0, 0, 0] - 50.), .01)
        self.assertLessEqual(abs(IXYZ[0, 0, 2] - 50.), .01)

    def test_XYZ_to_sRGB_list(self):
        """
            Test conversion between a list of XYZs and Yxy
        """
        RGB = Convert.XYZ_to_sRGB(np.array([[50, 50, 50]]))
        # Luminance should be the same
        self.assertLessEqual(abs(RGB[0] - 60.239), .01)

    def test_XYZ_to_sRGB_image(self):
        """
            Test conversion between a list of XYZs and Yxy
        """
        image = np.ones((2, 2, 3)) * 50
        image_rgb = Convert.XYZ_to_sRGB(image)
        # Luminance should be the same
        self.assertLessEqual(abs(image_rgb[0, 0, 0] - 60.239), .01)

    def test_sRGB_to_XYZ_backforth(self):
        v = Convert.XYZ_to_Yxy(Convert.sRGB_to_XYZ(np.array([np.eye(3)])))
        ex = np.array([[[0.21267285, 0.63999999, 0.32999999], [0.71515217, 0.3, 0.6], [0.072175, 0.15, 0.06000001]]])
        self.assertLessEqual(np.sum(np.abs(v - ex)), .01)

    def test_XYZ_to_Yxy_image(self):
        """
            Test conversion between an image of XYZs and Yxy
        """
        image = np.ones((1, 1, 3))
        image[0, 0, 0] = 50
        image[0, 0, 1] = 50
        image[0, 0, 2] = 50

        converted = Convert.XYZ_to_Yxy(image)
        self.assertEqual(converted[:, :, 0], 50)
        self.assertLessEqual(np.abs(converted[:, :, 1] - .3333), .0001)
        self.assertLessEqual(np.abs(converted[:, :, 2] - .3333), .0001)

    def test_kelvin_5000_to_Yxy(self):
        """
            Test approximation of plankian radiators without adaptation or refractive index
            Error +- .001
        """
        spectra_5000k = self.illuminants.get("B5000K")
        X, Y, Z = self.cmfs.get("CIE_1931_2deg")
        obs = np.array([[integrate(X, spectra_5000k), integrate(Y, spectra_5000k), integrate(Z, spectra_5000k)]])
        converted = Convert.XYZ_to_Yxy(obs)
        self.assertLessEqual(np.abs(converted[0][1] - 0.34510), .001)
        self.assertLessEqual(np.abs(converted[0][2] - 0.35161), .001)
