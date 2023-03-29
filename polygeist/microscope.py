import numpy as np
import polygeist.colour as pc


class Microscope:
    # The wavelength sensitivity range in nm that we will be concerned with
    # (just before the human visual range, to around 200nm into the infrared).
    wl = np.arange(300, 1100)

    # Calibration reported in the leica presentation given at the ICC.
    leica_r = pc.Illuminant.from_file_with_range("spectral/scanner/Leica_calibrated_r.csv", wl)
    leica_g = pc.Illuminant.from_file_with_range("spectral/scanner/Leica_calibrated_g.csv", wl)
    leica_b = pc.Illuminant.from_file_with_range("spectral/scanner/Leica_calibrated_b.csv", wl, interp="linear")

    # Raw functions of the sensor reported in the same presentation
    raw_leica_r = pc.Illuminant.from_file_with_range("spectral/raw_sensors//Leica_r.csv", wl)
    raw_leica_g = pc.Illuminant.from_file_with_range("spectral/raw_sensors//Leica_g.csv", wl)
    raw_leica_b = pc.Illuminant.from_file_with_range("spectral/raw_sensors//Leica_b.csv", wl)

    # Lightsource (INGAN)
    ls = pc.Illuminant.from_file_with_range("spectral/raw_sensors/LXA7-PW57.csv", wl)

    def response(self, filter_function, power=1., plot=False):
        """Calculate the response from the microscope given a filter function"""
        # illumination approximation (this can be substituted)
        filtered_function = (self.ls * power) * filter_function
        R = pc.integrate(filtered_function, self.raw_leica_r)
        G = pc.integrate(filtered_function, self.raw_leica_g)
        B = pc.integrate(filtered_function, self.raw_leica_b)

        if plot:
            from matplotlib import pyplot as plt
            plt.plot(self.wl, self.raw_leica_r * filtered_function, 'r')
            plt.plot(self.wl, self.raw_leica_g * filtered_function, 'g')
            plt.plot(self.wl, self.raw_leica_b * filtered_function, 'b')
            plt.xlabel("Wavelength nm")
            plt.ylabel("Response (relative to power)")
            plt.show()
        return R, G, B
