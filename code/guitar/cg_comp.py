import numpy as np
import pandas as pd

class GuitarString(object):

    '''Collect parameters and compute properties of guitar strings
       Estimate the frequency shift (due to frequency-pulling and dispersion)
       and the round-trip time delay (due to dispersion) of each mode q.
    
    Public Methods
    --------------
    set_scale : 
        Set the scale length of the string
    set_r :
        Set the response of the string's frequency to a change of it's length
    get_d_omega : numpy.ndarray.float64
        Frequency shift of each mode q relative to 2 * q * pi
    get_delta_omega : numpy.ndarray.float64
        Frequency of each mode q (relative to q = 0)
    get_omega : numpy.ndarray.float64
        Normalized frequency detuning for each mode q
    get_gamma : numpy.ndarray.complex128
        Complex ODE decay constant of each mode q
    get_delay : numpy.ndarray.float64
        Time delay prefactor for each mode q
    get_keywords : string
        String containing the attributes of a
        FrequencyShifts object
    '''

    def __init__(self, name, note, scale_length, diameter, linear_mass_density, tension, units='IPS'):
        '''Initialize a GuitarString object.

        Parameters
        ----------
        name : str
            A python string containing the name of the guitar string
        note : str
            A python string labeling the fundamental frequency of the
            open string using scientific notation, such as 'A_4', 'Ab_4',
            or 'A#_4'
        scale_length : numpy.float64
            The scale length of the guitar string in inches if units='IPS'
            or millimeters otherwise
        diameter : numpy.float64
            The diameter of the guitar string in inches if units='IPS'
            or millimeters otherwise
        linear_mass_density : numpy.float64
            The linear_mass_density of the guitar string in pounds per inch
            if units='IPS' or kilograms per mm otherwise
        tension : numpy.float64
            The nominal tension of the guitar string in pounds if units='IPS'
            or newtons otherwise
        units : str
            If units='IPS' (default), then the unit system of the input variables is
            assumed to use inches, pounds (for both mass and weight), and
            seconds; for any other value, the unit system of the input
            variables is assumed to use millimeters, kilograms, newtons, and
            seconds
        '''
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
        self._stiffness = np.sqrt( np.pi * (self._radius / 1000)**4 * modulus / ( 4 * self._tension * (self._scale / 1000)**2 ) )

    def set_scale(self, scale_length, units=None):
        if units == 'IPS':
            in_to_mm = 25.4
            scale_length *= in_to_mm
            
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
        '''Return a string displaying the attributes of a GuitarStrings object.
    
        Example
        -------
        strings = GuitarStrings(name, file_name, sheet_name, scale_length)
        
        print(strings)
        '''
        retstr = self._name + "\n"
        for string in self._strings:
            retstr += string.__str__() + "\n"
        return retstr

    def comp_r(self, delta_nu:list, g, b, c, ds, dn, x0):
        q = ( g / (2 * x0**2) ) * ( (b + c)**2 + b**2/(g - 1) - c**2/g )
        l0 = x0 + ds + dn

        kappa = np.zeros_like(l0)
        string_num = np.arange(0, len(self._strings))
        aa = q / 2
        for string, num in zip(self._strings, string_num):
            bb = string._radius / (2 * l0[num])
            cc = -(np.log(2)/1200.0) * delta_nu[num] - (g - 1) * (ds[num] - dn[num]) / x0
            kappa[num] = ( ( -bb + np.sqrt(bb**2 - 4 * aa * cc) ) / (2 * aa) )**2

        return (600.0 / np.log(2)) * (kappa + 1)
    
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
    def __init__(self, name, x0, dn, ds, b, c, string_count, strings, dbdx = 0.0, d=0.0):
        self._name = name
        self._x0 = x0
        if len(dn) == 1:
            self._dn = np.array(dn * string_count)
        else:
            self._dn = np.array(dn)
        if len(ds) == 1:
            self._ds = np.array(ds * string_count)
        else:
            self._ds = np.array(ds)
        self._b = b
        self._c = c
        self._dbdx = dbdx
        self._d = d
        
        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self.string_list = np.arange(1, string_count + 1)
        self._strings = strings
    
    def __str__(self):
        '''Return a string displaying the attributes of a Guitar object.

        Example
        -------
        guitar = Guitar(name, x0, dn, ds, b, c, string_count, strings)
        
        print(guitar)
        '''
        retstr = self._name + '\n'
        retstr += 'Scale Length: ' + '{:.1f} mm\n'.format(self._x0)
        if np.all(np.abs(self._ds - self._ds[0]) < 1.0e-06):
            retstr += 'Saddle Setback: ' + '{:.2f} mm\n'.format(self._ds[0])
        else:
            setback_str = 'Saddle Setback: ['
            template = '{:.{prec}}, '
            for m in np.arange(self._ds.size):
                setback_str += template.format(self._ds[m], prec = 3)
            retstr += setback_str[0:-2] + ']\n'
        if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
            retstr += 'Nut Setback: ' + '{:.2f} mm\n'.format(self._dn[0])
        else:
            setback_str = 'Nut Setback: ['
            template = '{:.{prec}}, '
            for m in np.arange(self._dn.size):
                setback_str += template.format(self._dn[m], prec = 2)
            retstr += setback_str[0:-2] + ']\n'
        retstr += 'b: ' + '{:.1f} mm\n'.format(self._b)
        retstr += 'c: ' + '{:.1f} mm\n'.format(self._c)
        retstr += 'dbdx: ' + '{:.1f}\n'.format(self._dbdx)
        retstr += 'd: ' + '{:.1f} mm\n'.format(self._d)
        retstr += self._strings.__str__() + "\n"
        return retstr
    
    def _bn(self, n):
        g_n = self.gamma(n)
        return self._b - ((g_n - 1)/g_n) * self._dbdx * self._x0

    def gamma(self, n):
        return 2.0**(n/12.0)

    def l0(self):
        length = np.sqrt( (self._x0 + self._ds + self._dn)**2 + self._c**2 )
        return length

    def l(self, fret_list):
        ds, n = np.meshgrid(self._ds, fret_list, sparse=False, indexing='ij')
        #length = np.sqrt( (self._x0/self.gamma(n) + ds)**2 + (self._b + self._c)**2 )
        length = np.sqrt( (self._x0/self.gamma(n) + ds)**2 + (self._bn(n) + self._c)**2 )
        return length

    def lp(self, fret_list):
        dn, n = np.meshgrid(self._dn, fret_list, sparse=False, indexing='ij')
        length = self._d + np.sqrt( (dn + self._x0 - self._x0/self.gamma(n) - self._d)**2 + self._bn(n)**2 )
        return length

    def lmc(self, fret_list):
        length = self.l(fret_list) + self.lp(fret_list)
        return length
    
    def qn(self, fret_list):
        l0 = np.tile(self.l0().reshape(-1, 1), (1, fret_list.size))
        return (self.lmc(fret_list) - l0) / l0
    
    def freq_shifts(self, fret_list):
        l_0 = np.tile(self.l0().reshape(-1, 1), (1, fret_list.size))
        kappa, n_2d = np.meshgrid(self._strings.get_kappa(), fret_list, sparse=False, indexing='ij')
        b_0, n_2d = np.meshgrid(self._strings.get_stiffness(), fret_list, sparse=False, indexing='ij')
        
        rle = 1200 * np.log2( l_0 / (self.gamma(n_2d) * self.l(fret_list)) )
        tme = (600/np.log(2.0)) * kappa * self.qn(fret_list)
        bse = 1200 * np.log2( (1 + self.gamma(n_2d) * b_0) / (1 + b_0) )
        
        shifts = rle + tme + bse
        
        open_strings = np.array([np.zeros(self.string_list[-1])]).T
        
        return np.hstack((open_strings, shifts))
    
    def comp_r(self, delta_nu:list, fret:int):
        g = self.gamma(fret)
        b = self._bn(fret)
        r = self._strings.comp_r(delta_nu, g, b, self._c, self._ds, self._dn, self._x0)
        return r
