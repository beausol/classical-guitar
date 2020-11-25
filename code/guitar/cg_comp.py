import numpy as np
import pandas as pd

class GuitarString(object):
    def __init__(self, name, note, scale_length, diameter, linear_mass_density, tension, units='IPS'):
        if units == 'IPS':
            in_to_mm = 25.4
            lb_to_kg = 1.0 / 2.204
            lb_to_nt = 9.81 / 2.204
            
            scale_length *= in_to_mm
            diameter *= in_to_mm
            linear_mass_density *= (lb_to_kg/in_to_mm)
            tension *= lb_to_nt

        self._name = name
        self._note = note
        self._freq = self._frequency(self._note)
        self._scale = scale_length
        self._radius = diameter / 2
        self._density_lin = linear_mass_density
        self._density_vol = self._density_lin / (np.pi * self._radius**2)
        self._tension = tension

    def __str__(self):
        '''Return a string displaying the attributes of a GuitarString object.

        Example
        -------
        string = GuitarString(name, note, scale_length, diameter, linear_mass_density, tension)
        
        print(string)
        '''
        retstr = self._name + " (" + self._note + " = {:5.1f} Hz) -- ".format(self._freq)
        retstr += 'Scale Length: ' + '{:.1f} mm; '.format(self._scale)
        retstr += 'Radius: ' + '{:.3f} mm; '.format(self._radius)
        retstr += 'Density: ' + '{:.3e} kg/mm; '.format(self._density_lin)
        retstr += 'Tension: ' + '{:.1f} N'.format(self._tension)
        return retstr
    
    def _frequency(self, note_str):
        notes = dict([('Ab', 49), ('A', 48), ('A#', 47), ('Bb', 47), ('B', 46), ('B#', 57), ('Cb', 46), ('C', 57), ('C#', 56),
                  ('Db', 56), ('D', 55), ('D#', 54), ('Eb', 54), ('E', 53), ('E#', 52), ('Fb', 53), ('F', 52), ('F#', 51),
                  ('Gb', 51), ('G', 50), ('G#', 49)])

        note = note_str.split('_')
        return 440.0 * 2**( int(note[1], 10) - notes[note[0]]/12.0 )

    def _comp_tension(self):
        mu = self._density_lin * 1000    # Convert kg/mm to kg/m
        x0 = self._scale / 1000      # Convert mm to m
        return mu * (2 * x0 * self._freq)**2

    def _comp_kappa(self):
        self._kappa = (np.log(2) / 600) * self._r - 1

    def _comp_modulus(self):
        self._modulus = 1.0e-09 * (self._tension / (np.pi * (self._radius/1000)**2)) * self._kappa

    def _comp_stiffness(self):
        modulus = 1.0e+09 * self._modulus
        self._stiffness = np.sqrt( (np.pi * (self._radius/1000)**4) * modulus / (self._tension * (self._scale / 1000)**2) )

    def set_scale(self, scale_length):
        self._scale = scale_length
        self._tension = self._comp_tension()
        
    def set_r(self, r):
        self._r = r
        self._comp_kappa()
        self._comp_modulus()
        self._comp_stiffness()
    
    def get_tension(self):
        return self._tension
    
    def get_density(self):
        return self._density_vol
    
    def get_kappa(self):
        return self._kappa
    
    def get_modulus(self):
        return self._modulus
    
    def get_stiffness(self):
        return self._stiffness


class GuitarStrings(object):
    def __init__(self, name, file_name, sheet_name=None, scale_length=None):
        self._name = name
        self._scale = scale_length
        if sheet_name is None:
            data = pd.read_excel(file_name,
                                 dtype={'name': str, 'note': str, 'scale': float,
                                        'diameter': float, 'density': float, 'tension': float})
        else:
            data = pd.read_excel(file_name, sheet_name=sheet_name,
                                 dtype={'name': str, 'note': str, 'scale': float,
                                        'diameter' : float, 'density': float, 'tension': float})

        self._strings = []
        row_list = np.arange(data.shape[0])
        for row in row_list:
            string = GuitarString(data.name[row], data.note[row], data.scale[row],
                            data.diameter[row], data.density[row], data.tension[row])
            if scale_length is not None:
                string.set_scale(scale_length)
            self._strings.append(string)

    def __str__(self):
        retstr = self._name + "\n"
        for string in self._strings:
            retstr += string.__str__() + "\n"
        return retstr

    def set_scale(self, scale_length):
        self._scale = scale_length
        for string in self._strings:
            string.set_scale(scale_length)

    def set_r(self, r):
        for string, r_string in zip(self._strings, r):
            string.set_r(r_string)

    def get_count(self):
        return len(self._strings)
    
    def get_tensions(self):
        tensions = []
        for string in self._strings:
            tensions.append(string.get_tension())
        return np.array(tensions)
    
    def get_densities(self):
        densities = []
        for string in self._strings:
            densities.append(string.get_density())
        return np.array(densities)
    
    def get_kappa(self):
        kappa = []
        for string in self._strings:
            kappa.append(string.get_kappa())
        return np.array(kappa)
    
    def get_modulus(self):
        modulus = []
        for string in self._strings:
            modulus.append(string.get_modulus())
        return np.array(modulus)
    
    def get_stiffness(self):
        stiffness = []
        for string in self._strings:
            stiffness.append(string.get_stiffness())
        return np.array(stiffness)


class Guitar(object):
    def __init__(self, name, x0, dn, ds, b, c, string_count, strings):
        self._name = name
        self._x0 = x0
        self._dn = dn
        self._ds = ds
        self._b = b
        self._c = c
        
        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self.string_list = np.arange(1, string_count + 1)
        self._strings = strings
    
    def __str__(self):
        '''Return a string displaying the attributes of a ModeLockedLaserModel object.

        Example
        -------
        model = ModeLockedLaserModel(params, freq_shifts, amplifier, absorber)
        
        print(model)
        '''
        retstr = self._name + '\n'
        retstr += 'Scale Length: ' + '{:.1f} mm\n'.format(self._x0)
        retstr += 'Nut Setback: ' + '{:.2f} mm\n'.format(self._dn)
        retstr += 'Saddle Setback: ' + '{:.2f} mm\n'.format(self._ds)
        retstr += 'b: ' + '{:.1f} mm\n'.format(self._b)
        retstr += 'c: ' + '{:.1f} mm\n'.format(self._c)
        retstr += self._strings.__str__() + "\n"
        return retstr
    
    def gamma(self, n):
        return 2.0**(n/12.0)

    def l(self, n):
        if n != 0:
            length = np.sqrt( (self._x0/self.gamma(n) + self._ds)**2 + (self._b + self._c)**2 )
        else:
            length = np.sqrt( (self._x0 + self._ds + self._dn)**2 + self._c**2 )
        return length

    def lp(self, n):
        if n != 0:
            length = np.sqrt( (self._dn + self._x0 - self._x0/self.gamma(n))**2 + self._b**2 )
        else:
            length = 0.0
        return length

    def lmc(self, n):
        length = self.l(n) + self.lp(n)
        return length
    
    def qn(self, n):
        l0 = self.l(0)
        return (self.lmc(n) - l0) / l0
