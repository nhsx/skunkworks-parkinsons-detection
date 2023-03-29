import polygeist.colour as pc
import numpy as np


class VirtualColourimetricScope:
    """
        Non-physically (Not path tracing) based model of an imaging device.
    """
    def __init__(self, filename_sensor_r, filename_sensor_g, filename_sensor_b, filename_illumination,
                 wl=np.arange(300, 1100)):
        """
        Constructor for Virtual Scope, builds sensor and illumination functions required to decompose
        RGB images
        @arg filename_sensor_r: Sensor sensitivity for R, as CSV. used with polygeist.colour.Illuminant (string)
        @arg filename_sensor_g: Sensor sensitivity for G, as CSV. used with polygeist.colour.Illuminant (string)
        @arg filename_sensor_b: Sensor sensitivity for B, as CSV. used with polygeist.colour.Illuminant (string)
        @arg filename_illumination: Light source for scope, as CSV. used with polygeist.colour.Illuminant (string)
        @arg wl: Wavelength as a range from start to finish in nm (np.array)
        @return: Object (VirtualColourimetricScope)
        """
        self.wl = wl
        self.r_function = pc.Illuminant.from_file_with_range(filename_sensor_r, wl)
        self.g_function = pc.Illuminant.from_file_with_range(filename_sensor_g, wl)
        self.b_function = pc.Illuminant.from_file_with_range(filename_sensor_b, wl)
        self.ls = pc.Illuminant.from_file_with_range(filename_illumination, wl)

    def response(self, filter_function, gamma=1.):
        """
        Produce a tristimulus response for the virtual scope, given a filter function
        @arg filter_function: Transmission or reflectance function, as a function of self.wl  (np.array)
        @arg gamma: Brightness of the light source in intensity, default = 1. (float)
        @return: (r, g, b) Tristimulus values from the scope (Tuple, float)
        """

        # Transmission as a function of wavelength
        # R(λ) = γI(λ)S(λ), gamma (γ) adjusts brightness of lightsource (I), surface (S).
        transmission = (self.ls * gamma) * filter_function

        # RGB = ∫R(λ)X[r,g,b](λ) dλ.  Sensor response is integration over the transmission by sensitivity
        r = pc.integrate(transmission, self.r_function)
        g = pc.integrate(transmission, self.g_function)
        b = pc.integrate(transmission, self.b_function)

        # return tristimulus values
        return r, g, b




