import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.optimize import curve_fit, minimize
import sys
import os

# Define a reasonable set of matplotlib parameters compatible
#  with Jupyter notebooks
labelsize = 18
fontsize = 24
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : fontsize,
        }
bbox = dict(boxstyle='round', edgecolor='#D7D7D7', facecolor='white', alpha=0.75)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')


def file_path(pathname, filename):
    '''
    Return a fully qualified path to a named file.
    
    Parameters
    ----------
    pathname : str
        A valid path to a directory / folder, or None.
    filename : str
        A va;lid file name (including an extension where needed),
        or None.
    
    Returns
    ----------
    retval : str
        If neither pathname nor filename is None, then
        pathname + filename; otherwise None.
    '''
    if (pathname is None) or (filename is None):
        return None
    else:
        return pathname + filename


def get_xlim():
    '''
    Return the "nice" values of the min and max of the x-axis of the current plot
    for matplotlib graphics.
    
    Returns
    ----------
    retval : numpy.ndarray.float64
        The minimum [0] and maximum [1] of the x values, rounded to a value that's
        convenient for a matplotlib graph.
    '''
    loc = plt.xticks()[0]
    retval =  np.array([loc[0], loc[len(loc)-1]])

    return retval

def get_ylim():
    '''
    Return the "nice" values of the min and max of the y-axis of the current plot
    for matplotlib graphics.
    
    Returns
    ----------
    retval : numpy.ndarray.float64
        The minimum [0] and maximum [1] of the y values, rounded to a value that's
        convenient for a matplotlib graph.
    '''
    loc = plt.yticks()[0]
    retval =  np.array([loc[0], loc[len(loc)-1]])

    return retval


class GuitarString(object):
    '''Collect parameters and compute properties of guitar strings
       Estimate the frequency shift (due to frequency-pulling and dispersion)
       and the round-trip time delay (due to dispersion) of each mode q.
    
    Public Methods
    --------------
    set_scale_length : 
        Set the scale length of the string
    set_r :
        Set the response of the string's frequency to a change of its length
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

    def __init__(self, params, props, units='IPS'):
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
            The scale length of the guitar string (2x the distance measured
            from the inside edge of the nut to the center of the twelfth
            fret) in inches if units='IPS' or millimeters otherwise
        diameter : numpy.float64
            The diameter of the guitar string in inches if units='IPS'
            or millimeters otherwise
        linear_mass_density : numpy.float64
            The linear_mass_density of the guitar string in pounds per inch
            if units='IPS' or milligrams per millimeter otherwise
        tension : numpy.float64
            The nominal tension of the guitar string in pounds if units='IPS'
            or newtons otherwise
        units : str
            If units='IPS' (default), then the unit system of the input variables is
            assumed to use inches, pounds (for both mass and weight), and
            seconds; for any other value, the unit system of the input
            variables is assumed to use millimeters, milligrams, newtons, and
            seconds
        '''
        self._params = params
        self._props = props
        
        if units == 'IPS':
            in_to_mm = 25.4
            lb_to_mg = 453592.37
            lb_to_nt = 4.4482216153 # 9.81 / 2.204
            
            self._params.scale_length *= in_to_mm
            self._params.diameter *= in_to_mm
            self._params.linear_mass_density *= (lb_to_mg/in_to_mm)
            self._params.tension *= lb_to_nt

        freq = self._frequency(self._params.note)
        radius = self._params.diameter / 2
        params = pd.concat([[params, pd.Series({'freq':freq}), pd.Series({'radius':radius})]])

    def __str__(self):
        '''Return a string displaying the attributes of a GuitarString object.

        Example
        -------
        string = GuitarString(name, note, scale_length, diameter, linear_mass_density, tension)
        
        print(string)
        '''
        retstr = self._name + " (" + self._note + " = {:5.1f} Hz) -- ".format(self._freq)
        retstr += 'Scale Length: ' + '{} mm; '.format(round(self._scale_length))
        retstr += 'Radius: ' + '{:.3f} mm; '.format(self._radius)
        retstr += 'Density: ' + '{:.3f} mg/mm; '.format(self._density_lin)
        retstr += 'Tension: ' + '{:.1f} N'.format(self._tension)
        return retstr
    
    def _frequency(self, note_str):
        notes = dict([('Ab', 49), ('A', 48), ('A#', 47), ('Bb', 47), ('B', 46), ('B#', 57), ('Cb', 46), ('C', 57), ('C#', 56),
                  ('Db', 56), ('D', 55), ('D#', 54), ('Eb', 54), ('E', 53), ('E#', 52), ('Fb', 53), ('F', 52), ('F#', 51),
                  ('Gb', 51), ('G', 50), ('G#', 49)])

        note = note_str.split('_')
        return 440.0 * 2**( int(note[1], 10) - notes[note[0]]/12.0 )

    def _comp_tension(self):
        mu = self._density_lin / 1000       # Convert mg/mm to kg/m
        x0 = self._scale_length / 1000      # Convert mm to m
        self._tension = mu * (2 * x0 * self._freq)**2

    def _comp_kappa(self):
        self._kappa = 2 * self._r + 1

    def _comp_stiffness_old(self):
        modulus = 1.0e+09 * self._modulus
        self._stiffness = np.sqrt( np.pi * (self._radius / 1000)**4 * modulus / ( 4 * self._tension * (self._scale_length / 1000)**2 ) )

    def _comp_stiffness(self):
        self._stiffness = np.sqrt(self._kappa) * (self._radius / 1000) / ( 2 * self._scale_length / 1000)

    def _comp_modulus(self):
        self._modulus = 1.0e-09 * (self._tension / (np.pi * (self._radius/1000)**2)) * self._kappa

    def set_scale_length(self, scale_length, units=None):
        if units == 'IPS':
            in_to_mm = 25.4
            scale_length *= in_to_mm
            
        self._scale_length = scale_length
        self._comp_tension()
        
    def fit_r(self, dx, df, scale):
        def func(x, intercept, slope):
            return intercept + slope * x

        param, param_cov = curve_fit(func, scale*dx, df)
        fit = func(scale*dx, *param)
        
        dfdx = param[1]
        ddfdx = np.sqrt(param_cov[1][1])
        
        r = (self._scale_length / self._freq) * dfdx
        dr = (self._scale_length / self._freq) * ddfdx
        
        self.set_r(r, dr)

        return fit
    
    def set_r(self, r, dr):
        self._r = r
        self._dr = dr
        self._comp_kappa()
        self._comp_stiffness()
        self._comp_modulus()
    
    def get_radius(self):
        return self._radius
    
    def get_tension(self):
        return self._tension
    
    def get_density(self):
        return self._density_lin
    
    def get_density_vol(self):
        return self._density_vol
    
    def get_name(self):
        return self._name
    
    def get_note(self):
        return self._note
    
    def get_r(self):
        return (self._r, self._dr)
    
    def get_kappa(self):
        return self._kappa
    
    def get_modulus(self):
        return self._modulus
    
    def get_stiffness(self):
        return self._stiffness


class GuitarStrings(object):
    def __init__(self, name, file_name, sheet_name=None, scale_length=None):
        self._name = name
        self._scale_length = scale_length
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
                string.set_scale_length(scale_length)
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
    
    def set_scale_length(self, scale_length):
        self._scale_length = scale_length
        for string in self._strings:
            string.set_scale_length(scale_length)

    def fit_r(self, datapath, sheet_name=0, scale=1.0, show=True, savepath=None, filename=None, markersize=12.5):
        data = pd.read_excel(datapath, sheet_name=sheet_name)
        column_names = list(data.columns)
        dx = np.array(data[[column_names[0]]].values.T[0])
        self.compare_string_names(column_names[1:])
        
        fit_dict = {}
        for string in self._strings:
            name = string.get_name()
            fit = string.fit_r(dx, data[name].values -  data[name].values[0], scale)
            fit_dict[name] = fit
        
        self.plot_fit(fit_dict, data, show, savepath, filename, markersize)

    def plot_fit(self, fit_dict, data, show, savepath, filename, markersize):
        dx = np.array(data[[list(data.columns)[0]]].values.T[0])
        
        plt.figure(figsize=(8.0,6.0))

        for string in self._strings:
            name = string.get_name()
            plt.plot(dx, data[name].values -  data[name].values[0], '.', markersize=markersize)
            plt.plot(dx, fit_dict[name], color=plt.gca().lines[-1].get_color(), label='{}'.format(name))

        plt.xlabel(r'$\Delta x$~(mm)', fontdict=font)
        plt.ylabel(r'$\Delta f$~(Hz)', fontdict=font)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', labelsize=labelsize)
        plt.xlim(dx[0], dx[-1])
        plt.ylim(0, get_ylim()[1])
        plt.legend(loc='upper left', fontsize=labelsize)
        plt.grid(True)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            plt.savefig(filepath, bbox_inches='tight')
            print("Saved {0}\n".format(filepath))
        if show:
            plt.show()
        else:
            plt.close()

    def get_count(self):
        return len(self._strings)
    
    def get_string_names(self):
        names = []
        for string in self._strings:
            names.append(string.get_name())
        return names

    def compare_string_names(self, name_list):
        string_names = self.get_string_names()
        assert len(string_names) == len(name_list), 'This string set has {} strings, not {}.'.format(len(string_names), len(name_list))
        assert sorted(string_names) == sorted(name_list), 'The input string names should be {}, not {}.'.format(string_names, name_list)
    
    def get_notes(self):
        notes = []
        for string in self._strings:
            note_str = string.get_note()
            note = note_str.split('_')
            notes.append(note[0] + '$_{' + note[1] + '}$')
        return notes

    def get_radii(self):
        radii = []
        for string in self._strings:
            radii.append(string.get_radius())
        return np.array(radii)

    def get_densities(self):
        densities = []
        for string in self._strings:
            densities.append(string.get_density())
        return np.array(densities)
    
    def get_tensions(self):
        tensions = []
        for string in self._strings:
            tensions.append(string.get_tension())
        return np.array(tensions)

    def get_r(self):
        r = []
        dr = []
        for string in self._strings:
            r.append(string.get_r()[0])
            dr.append(string.get_r()[1])
        return np.array(r), np.array(dr)

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

    def compensate(self, g_n, q_n, s):
        def sigma_n(g_n, k):
            return np.sum((g_n - 1)**k)

        sigma_0 = sigma_n(g_n, 0)
        sigma_1 = sigma_n(g_n, 1)
        sigma_2 = sigma_n(g_n, 2)
        sigma_3 = sigma_n(g_n, 3)
        
        sigma = np.array([[sigma_2, -sigma_1], [sigma_1, -sigma_0]])
        sum_qn = 0.5 * np.sum(q_n)
        sum_gq = 0.5 * np.sum((g_n - 1) * q_n)

        idx_list = np.arange(0, self.get_count())
        kappa = self.get_kappa()
        b0 = self.get_stiffness() * s
        ds = np.zeros(self.get_count())
        dn = np.zeros(self.get_count())
#        for string, idx in zip(self._strings, idx_list):
        for idx in idx_list:
            b_1 = sigma_1 * b0[idx] + 0.5 * (1 + np.pi**2) * (sigma_2 + 2.0 * sigma_1) * b0[idx]**2
            b_2 = sigma_2 * b0[idx] + 0.5 * (1 + np.pi**2) * (sigma_3 + 2.0 * sigma_2) * b0[idx]**2
            rhs = np.array([[b_2 + sum_gq * kappa[idx]], [b_1 + sum_qn * kappa[idx]]])
            lhs = np.dot(np.linalg.inv(sigma), rhs)
            ds[idx] = lhs[0]
            dn[idx] = lhs[1]
        
        return ds, dn

    def save_specs_table(self, show=True, savepath=None, filename=None):
        names = self.get_string_names()
        notes = self.get_notes()
        radii = self.get_radii()
        densities = self.get_densities()
        tensions = self.get_tensions()

        df = pd.DataFrame({'String': names,
                           'Note': notes,
                           'Radius (mm)': radii.tolist(),
                           'Density (mg/mm)': densities.tolist(),
                           'Tension (N)': tensions.tolist()})
        
        formatter = {'Radius (mm)': '{:.3f}',
                     'Density (mg/mm)': '{:.3f}',
                     'Tension (N)': '{:.1f}'}

        styler = df.style.format(formatter=formatter).hide()
        table_str = styler.to_latex(column_format='cccccc', hrules=True)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            print(table_str,  file=open(filepath, 'w'))        
            print("Saved {0}\n".format(filepath))

        if show:
            styler.set_properties(**{'text-align': 'center'})
            display(styler)
 
    def save_props_table(self, show=True, savepath=None, filename=None):
        names = self.get_string_names()
        r, dr = self.get_r()
        kappa = self.get_kappa()
        modulus = self.get_modulus()
        stiffness = self.get_stiffness()

        df = pd.DataFrame({'String': names,
                           '$R$': r.tolist(),
                           '$\sigma$': dr.tolist(),
                           '$\kappa$': kappa.tolist(),
                           '$B_0$': stiffness.tolist(),
                           '$E$ (GPa)': modulus.tolist()})

        formatter = {'$R$': '{:.1f}',
                     '$\sigma$': '{:.1f}',
                     '$\kappa$': '{:.1f}',
                     '$B_0$': '{:.5f}',
                     '$E$ (GPa)': '{:.2f}'}
 
        styler = df.style.format(formatter=formatter).hide()
        table_str = styler.to_latex(column_format='cccccc', hrules=True)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            print(table_str,  file=open(filepath, 'w'))        
            print("Saved {0}\n".format(filepath))

        if show:
            styler.set_properties(**{'text-align': 'center'})
            display(styler)
 
    def save_props_excel(self, filepath, sheet_name):
        names = self.get_string_names()
        r, dr = self.get_r()
        kappa = self.get_kappa()
        modulus = self.get_modulus()
        stiffness = self.get_stiffness()

        df = pd.DataFrame({'String': names,
                           'R': r.tolist(),
                           'sigma': dr.tolist(),
                           'kappa': kappa.tolist(),
                           'B_0': stiffness.tolist(),
                           'E': modulus.tolist()})

        if os.path.isfile(filepath):
            with pd.ExcelWriter(filepath,  mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        else:
            df.to_excel(filepath, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        print("\nSaved {} : {}\n".format(filepath, sheet_name))
        

class Guitar(object):
    def __init__(self, name, string_count, strings, x0, ds, dn, b, c, d=0.0, rgx=1.0):
        self._name = name

        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self._string_list = np.arange(1, string_count + 1)
        self._strings = strings
    
        # self._x0 = x0

        # self.setbacks(ds, dn)

        # self._b = self._setarr(b)
        # self._c = self._setarr(c)
        # self._d = d
        
        # self._rgx = self._setarr(rgx)
        
        self.set_vars(
            x0 = x0,
            ds = ds,
            dn = dn,
            b = b,
            c = c,
            d = d,
            rgx = rgx
        )
        
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
            retstr += 'Saddle Setback: ' + np.array2string(self._ds[0], precision=2, floatmode='fixed', separator=', ') + ' mm\n'
            # setback_str = 'Saddle Setback: ['
            # template = '{:.{prec}}, '
            # for m in np.arange(self._ds.size):
            #     setback_str += template.format(self._ds[m], prec = 3)
            # retstr += setback_str[0:-2] + ']\n'
        if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
            retstr += 'Nut Setback: ' + '{:.2f} mm\n'.format(self._dn[0])
        else:
            retstr += 'Nut Setback: ' + np.array2string(self._dn[0], precision=2, floatmode='fixed', separator=', ') + ' mm\n'
            # setback_str = 'Nut Setback: ['
            # template = '{:.{prec}}, '
            # for m in np.arange(self._dn.size):
            #     setback_str += template.format(self._dn[m], prec = 2)
            # retstr += setback_str[0:-2] + ']\n'
        if np.all(np.abs(self._b - self._b[0]) < 1.0e-06):
            retstr += 'b: ' + '{:.2f} mm\n'.format(self._b[0])
        else:
            retstr += 'b: ' + np.array2string(self._b, precision=2, floatmode='fixed', separator=', ') + ' mm\n'
        # retstr += 'c: ' + '{:.1f} mm\n'.format(self._c)
        # retstr += 'd: ' + '{:.1f} mm\n'.format(self._d)
        if np.all(np.abs(self._c - self._c[0]) < 1.0e-06):
            retstr += 'c: ' + '{:.2f} mm\n'.format(self._c[0])
        else:
            retstr += 'c: ' + np.array2string(self._c, precision=2, floatmode='fixed', separator=', ') + ' mm\n'
        retstr += 'd: ' + '{:.1f} mm\n'.format(self._d)
        if np.all(np.abs(self._rgx - self._rgx[0]) < 1.0e-06):
            retstr += 's: ' + '{:.2f}\n'.format(self._rgx[0])
        else:
            retstr += 's: ' + np.array2string(self._rgx, precision=2, floatmode='fixed', separator=', ') + '\n'
        retstr += self._strings.__str__() + "\n"
        # rmsstr = 'RMS Frequency Errors: ['#+ '{:.2f} cents\n'.format(self._rms())
        # template = '{:.{prec}}, '
        # rms = self._rms()
        # for m in np.arange(rms.size):
        #     rmsstr += template.format(rms[m], prec = 2)
        # retstr += rmsstr[0:-2] + ']\n'

        return retstr
    
    # def _bn(self, n):
    #     g_n = self.gamma(n)
    #     return self._b - ((g_n - 1)/g_n) * self._dbdx * self._x0
    def _setarr(self, x):
        string_count = self._strings.get_count()
        if isinstance(x, np.ndarray):
            assert x.shape == (string_count,), 'Input array has shape {}, not {}.'.format(x.shape, (string_count,))
            return x
        elif isinstance(x, list):
            if len(x) == 1:
                return np.array(string_count * x)
            else:
                assert len(x) == string_count, 'Input list has length {}, not {}.'.format(len(x), string_count)
                return np.array(x)
        else:
            return np.array(string_count * [x])

    def _tile_strings(self, x):
        return np.tile(x, (self._strings.get_count(), 1))
    
    def _tile_frets(self, x, fret_list):
        return np.tile(x, (fret_list.size, 1)).T
    
    def _rms(self, dnu):
        rms = np.sqrt(np.mean(dnu**2))
        return rms

    def set_vars(self, **kwargs):
        x0 = kwargs.get('x0')
        if x0 is None:
            pass
        else:
            self._x0 = x0
        
        ds = kwargs.get('ds')
        if ds is None:
            pass
        else:
            self._ds = self._setarr(ds)

        dn = kwargs.get('dn')
        if dn is None:
            pass
        else:
            self._dn = self._setarr(dn)

        b = kwargs.get('b')
        if b is None:
            pass
        else:
            self._b = self._setarr(b)

        c = kwargs.get('c')
        if c is None:
            pass
        else:
            self._c = self._setarr(c)

        d = kwargs.get('d')
        if d is None:
            pass
        else:
            self._d = d

        rgx = kwargs.get('rgx')
        if rgx is None:
            pass
        else:
            self._rgx = self._setarr(rgx)

    def _gamma(self, n):
        return 2.0**(n/12.0)

    # def l0(self):
    #     length = np.sqrt( (self._x0 + self._ds + self._dn)**2 + self._c**2 )
    #     return length

    def _l0(self, ds, dn, x0, c):
        length = np.sqrt( (x0 + ds + dn)**2 + c**2 )
        return length

#     def l(self, fret_list):
# #        ds, n = np.meshgrid(self._ds, fret_list, sparse=False, indexing='ij')
#         ds = self._tile_frets(self._ds, fret_list)
#         b = self._tile_frets(self._b, fret_list)
#         c = self._tile_frets(self._c, fret_list)
#         n = self._tile_strings(fret_list)
# #        length = np.sqrt( (self._x0/self.gamma(n) + ds)**2 + (self._b + self._c)**2 )
#         length = np.sqrt( (self._x0/self._gamma(n) + ds)**2 + (b + c)**2 )
#         return length

    def _l(self, ds, x0, b, c, n):
        length = np.sqrt( (x0/self._gamma(n) + ds)**2 + (b + c)**2 )
        return length

    def _lp(self, ds, dn, x0, b, c, d, n):
        xn = x0 / self._gamma(n)
        ln = self._l(ds, x0, b, c, n)
        length = np.sqrt( (x0 - xn + dn - d)**2 + (b + (b + c) * d / (xn + ds))**2 )
        length += ( ln / (xn + ds)  ) * d
        return length

    # def lp(self, fret_list):
    #     x0 = self._x0
    #     ds = self._tile_frets(self._ds, fret_list)
    #     dn = self._tile_frets(self._dn, fret_list)
    #     b = self._tile_frets(self._b, fret_list)
    #     c = self._tile_frets(self._c, fret_list)
    #     d = self._d
    #     n = self._tile_strings(fret_list)
    #     xn = x0 / self._gamma(n)
    #     #length = np.sqrt( (dn + self._x0 - self._x0/self.gamma(n))**2 + self._bn(n)**2 )
    #     length = np.sqrt( (x0 - xn + dn - d)**2 + (b + (b + c) * d / (xn + ds))**2 )
    #     length += ( self.l(fret_list) / (xn + ds)  ) * d
    #     return length

    # def lp_old(self, fret_list):
    #     dn, n = np.meshgrid(self._dn, fret_list, sparse=False, indexing='ij')
    #     ds, n = np.meshgrid(self._ds, fret_list, sparse=False, indexing='ij')
    #     x0 = self._x0
    #     xn = x0 / self._gamma(n)
    #     b = self._b
    #     c = self._c
    #     d = self._d
    #     #length = np.sqrt( (dn + self._x0 - self._x0/self.gamma(n))**2 + self._bn(n)**2 )
    #     length = np.sqrt( (x0 - xn + dn - d)**2 + (b + (b + c) * d / (xn + ds))**2 )
    #     length += ( self.l(fret_list) / (xn + ds)  ) * d
    #     return length

    # def lmc(self, fret_list):
    #     length = self.l(fret_list) + self.lp(fret_list)
    #     return length
    
    def _lmc(self, ds, dn, x0, b, c, d, n):
        length = self._l(ds, x0, b, c, n) + self._lp(ds, dn, x0, b, c, d, n)
        return length
    
#     def qn(self, fret_list):
# #        l0 = np.tile(self.l0().reshape(-1, 1), (1, fret_list.size))
#         l0 = self._tile_frets(self.l0(), fret_list)
#         return (self.lmc(fret_list) - l0) / l0
    
    def _q(self, ds, dn, x0, b, c, d, n):
#        l0 = np.tile(self.l0().reshape(-1, 1), (1, fret_list.size))
        l0 = self._l0(ds, dn, x0, c)
        return (self._lmc(ds, dn, x0, b, c, d, n) - l0) / l0
    
    def _rle(self, ds, dn, x0, b, c, n):
        return 1200 * np.log2( self._l0(ds, dn, x0, c) / (self._gamma(n) * self._l(ds, x0, b, c, n)) )
    
    def _mde(self, ds, dn, x0, b, c, d, n):
        return 600 * np.log2( 1 + self._q(ds, dn, x0, b, c, d, n) )
    
    def _tse(self, ds, dn, x0, b, c, d, kappa, n):
        return 600 * np.log2( 1 + kappa * self._q(ds, dn, x0, b, c, d, n) )

    def _bse(self, b_0, n):
        return 1200 * np.log2( (1 + self._gamma(n) * b_0 + (1.0 + 0.5 * np.pi**2) * (self._gamma(n) * b_0)**2)
                             / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
        
    def _tfe(self, ds, dn, x0, b, c, d, kappa, b_0, n):
        rle = self._rle(ds, dn, x0, b, c, n)
        mde = self._mde(ds, dn, x0, b, c, d, n)
        tse = self._tse(ds, dn, x0, b, c, d, kappa, n)
        bse = self._bse(b_0, n)
        
        return rle + mde + tse + bse

    def _freq_shifts(self, fret_list):
        x0 = self._x0
        ds = self._tile_frets(self._ds, fret_list)
        dn = self._tile_frets(self._dn, fret_list)
        b = self._tile_frets(self._b, fret_list)
        c = self._tile_frets(self._c, fret_list)
        d = self._d
        kappa = self._tile_frets(self._strings.get_kappa(), fret_list)
        b_0 = self._tile_frets(self._strings.get_stiffness() * self._rgx , fret_list)
        n = self._tile_strings(fret_list)
        
        shifts = self._tfe(ds, dn, x0, b, c, d, kappa, b_0, n)
        
        open_strings = np.array([np.zeros(self._string_list[-1])]).T
        
        return np.hstack((open_strings, shifts))
    
    # def freq_shifts_old(self, fret_list):
    #     l_0 = self._tile_frets(self.l0(), fret_list)
    #     kappa = self._tile_frets(self._strings.get_kappa(), fret_list)
    #     b_0 = self._tile_frets(self._strings.get_stiffness() * self._rgx , fret_list)
    #     n = self._tile_strings(fret_list)
        
    #     # l_0 = np.tile(self.l0().reshape(-1, 1), (1, fret_list.size))
    #     # kappa, n_2d = np.meshgrid(self._strings.get_kappa(), fret_list, sparse=False, indexing='ij')
    #     # b_0, n_2d = np.meshgrid(self._strings.get_stiffness(), fret_list, sparse=False, indexing='ij')
        
    #     # rle = 1200 * np.log2( l_0 / (self.gamma(n_2d) * self.l(fret_list)) )
    #     # mde =  600 * np.log2( 1 + self.qn(fret_list) )
    #     # tse =  600 * np.log2( 1 + kappa * self.qn(fret_list) )
    #     # #bse = 1200 * np.log2( (1 + self.gamma(n_2d) * b_0) / (1 + b_0) )
    #     # bse = 1200 * np.log2( (1 + self.gamma(n_2d) * b_0 + (1.0 + 0.5 * np.pi**2) * (self.gamma(n_2d) * b_0)**2)
    #     #                       / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
    #     rle = 1200 * np.log2( l_0 / (self._gamma(n) * self.l(fret_list)) )
    #     mde =  600 * np.log2( 1 + self.qn(fret_list) )
    #     tse =  600 * np.log2( 1 + kappa * self.qn(fret_list) )
    #     bse = 1200 * np.log2( (1 + self._gamma(n) * b_0 + (1.0 + 0.5 * np.pi**2) * (self._gamma(n) * b_0)**2)
    #                         / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
        
    #     shifts = rle + mde + tse + bse
        
    #     open_strings = np.array([np.zeros(self._string_list[-1])]).T
        
    #     return np.hstack((open_strings, shifts))
    
    # def estimate_r(self, delta_nu:list, fret:int):
    #     g = self.gamma(fret)
    #     q = self.qn(np.array([fret]))[0,0]
    #     r = self._strings.estimate_r(delta_nu, g, q, self._ds, self._dn, self._x0)
    #     return r
    
    # def compensate_older(self, max_fret:int):
    #     self._ds = np.array([0.0] * self._strings.get_count())
    #     self._dn = np.array([0.0] * self._strings.get_count())
        
    #     fret_list = np.arange(1, max_fret + 1)
    #     g_n = self._gamma(fret_list)
    #     q_n = self.qn(fret_list)[0]

    #     ds, dn = self._strings.compensate(g_n, q_n, self._rgx)
    #     self._ds = self._x0 * ds
    #     self._dn = self._x0 * dn
        
    #     return self._ds, self._dn

    def approximate(self):
        x_0 = self._x0
        b = self._b
        c = self._c
        d = self._d
        kappa = self._strings.get_kappa()
        b_0 = self._strings.get_stiffness() * self._rgx

        fret_d = kappa * 2 * (2*b + c)**2 * d / x_0**2
        ds = b_0 * x_0 + kappa * ( (b + c)**2 - 7 * b**2 ) / (4 * x_0) - fret_d
        dn = -kappa * b * (5 * b + c)/(2 * x_0) - fret_d

        self._ds = ds
        self._dn = dn
        
        return ds, dn

    def compensate(self, max_fret:int):
        def sigma_k(fret_list, k):
            return np.sum((self._gamma(fret_list) - 1)**k)

        fret_list = np.arange(1, max_fret + 1)

        x0 = self._x0
        b = self._tile_frets(self._b, fret_list)
        c = self._tile_frets(self._c, fret_list)
        d = self._d
        kappa = self._tile_frets(self._strings.get_kappa(), fret_list)
        b_0 = self._tile_frets(self._strings.get_stiffness() * self._rgx , fret_list)
        n = self._tile_strings(fret_list)
        ds = np.zeros(n.shape)
        dn = np.zeros(n.shape)
        
        mde = self._mde(ds, dn, x0, b, c, d, n)
        tse = self._tse(ds, dn, x0, b, c, d, kappa, n)
        bse = self._bse(b_0, n)
        
        z_n = (np.log(2.0) / 1200.0) * ( mde + tse + bse )
        
        sigma_0 = max_fret
        sigma_1 = sigma_k(fret_list, 1)
        sigma_2 = sigma_k(fret_list, 2)
        
        zbar_0 = np.sum(z_n, axis=1)
        zbar_1 = np.sum((self._gamma(fret_list) - 1) * z_n, axis=1)
        
        det = sigma_0 * sigma_2 - sigma_1**2
        ds =  ( sigma_0 * zbar_1 - sigma_1 * zbar_0 ) * self._x0 / det
        dn = -( sigma_2 * zbar_0 - sigma_1 * zbar_1 ) * self._x0 / det

        self._ds = ds
        self._dn = dn
        
        return ds, dn

    def minimize(self, max_fret:int, approx:bool=False):
        def minfun(dsdn, *args):
            ds = dsdn[0]
            dn = dsdn[1]
            x0 = args[0]
            b = args[1]
            c = args[2]
            d = args[3]
            kappa = args[4]
            b_0 = args[5]
            n = args[6]
            
            tfe = self._tfe(ds, dn, x0, b, c, d, kappa, b_0, n)
            
            return np.sqrt(np.mean(tfe**2))
        
        fret_list = np.arange(1, max_fret + 1)

        if approx:
            ds_0, dn_0 = self.approximate()
        else:
            ds_0, dn_0 = self.compensate(max_fret)
        x0 = self._x0
        b = self._b
        c = self._c
        d = self._d
        kappa = self._strings.get_kappa()
        b_0 = self._strings.get_stiffness() * self._rgx

        ds = np.zeros_like(ds_0)
        dn = np.zeros_like(dn_0)
        idx_list = np.arange(0, self._strings.get_count())
        for idx in idx_list:
            dsdn0 = np.array([ds_0[idx], dn_0[idx]])
            args = (x0, b[idx], c[idx], d, kappa[idx], b_0[idx], fret_list)
            res = minimize(minfun, dsdn0, args)
            ds[idx] = res.x[0]
            dn[idx] = res.x[1]
            
        self._ds = ds
        self._dn = dn
    
        return ds, dn

    # def compensate_old(self, max_fret:int):
    #     def sigma_k(fret_list, k):
    #         return np.sum((self._gamma(fret_list) - 1)**k)

    #     fret_list = np.arange(1, max_fret + 1)

    #     n = self._tile_strings(fret_list)

    #     self._ds = np.zeros(n.shape)
    #     self._dn = np.zeros(n.shape)

    #     kappa = self._tile_frets(self._strings.get_kappa(), fret_list)
    #     b_0 = self._tile_frets(self._strings.get_stiffness() * self._rgx , fret_list)

    #     mde =  600 * np.log2( 1 + self.qn(fret_list) )
    #     tse =  600 * np.log2( 1 + kappa * self.qn(fret_list) )
    #     bse = 1200 * np.log2( (1 + self._gamma(n) * b_0 + (1.0 + 0.5 * np.pi**2) * (self._gamma(n) * b_0)**2)
    #                         / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
        
    #     z_n = mde + tse + bse
        
    #     sigma_0 = max_fret
    #     sigma_1 = sigma_k(fret_list, 1)
    #     sigma_2 = sigma_k(fret_list, 2)
        
    #     zbar_0 = np.sum(z_n, axis=1)
    #     zbar_1 = np.sum((self._gamma(fret_list) - 1) * z_n, axis=1)
        
    #     det = sigma_0 * sigma_2 - sigma_1**2
    #     ds =  ( sigma_0 * zbar_1 - sigma_1 * zbar_0 ) * self._x0 / det
    #     dn = -( sigma_2 * zbar_0 - sigma_1 * zbar_1 ) * self._x0 / det

    #     self._ds = ds
    #     self._dn = dn
        
    #     return ds, dn

    # def setbacks(self, ds, dn):
    #     string_count = self._strings.get_count()
    #     if len(ds) == 1:
    #         self._ds = np.array(ds * string_count)
    #     else:
    #         self._ds = np.array(ds)
    #     if len(dn) == 1:
    #         self._dn = np.array(dn * string_count)
    #     else:
    #         self._dn = np.array(dn)

    # def plot_shifts_older(self, max_fret=12, savepath=None, filename=None):
    #     fret_list = np.arange(0, max_fret + 1)
    #     shifts = self.freq_shifts(fret_list[1:])
    #     rms = np.sqrt(np.mean(shifts**2))
    #     names = self._strings.get_string_names()

    #     # labelsize = 18
    #     # fontsize = 24
    #     # font = {'family' : 'serif',
    #     #         'color'  : 'black',
    #     #         'weight' : 'normal',
    #     #         'size'   : fontsize,
    #     #         }
    #     # plt.rc('text', usetex=True)
    #     # plt.rc('font', family='serif')
        
    #     plt.figure(figsize=(8.0,6.0))
    #     for string in self._string_list:
    #         plt.plot(fret_list, shifts[string-1], label='{}'.format(names[string-1]))
    #     plt.xlabel('FRET', fontdict=font)
    #     plt.ylabel('SHIFT (cents)', fontdict=font)
    #     plt.tick_params(axis='x', labelsize=labelsize)
    #     plt.tick_params(axis='y', labelsize=labelsize)
    #     plt.xlim(fret_list[0],fret_list[-1])
    #     plt.ylim(-4,10)
    #     plt.legend(loc='upper right', fontsize=labelsize)
        
    #     if np.all(np.abs(self._ds - self._ds[0]) < 1.0e-06):
    #         ds = self._ds[0]
    #         template_ds = '$\Delta S = {}~\mathrm{{mm}}$'
    #     else:
    #         ds = np.round(np.mean(self._ds), 2)
    #         template_ds = '$\Delta S = {}~\mathrm{{mm~(mean)}}$'
    #     if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
    #         dn = self._dn[0]
    #         template_dn = '$\Delta N = {}~\mathrm{{mm}}$'
    #     else:
    #         dn = np.round(np.mean(self._dn), 2)
    #         template_dn = '$\Delta N = {}~\mathrm{{mm~(mean)}}$'
    #     template = '{}\n' + template_ds + '\n' + template_dn + '\n' + 'Shift~(rms):~{}~cents'
    #     plt.text(0.35, 5.3, template.format(self._name, ds, dn, np.round(rms, 2)),
    #              fontdict=font, bbox=bbox)
    #     plt.grid(True)
        
    #     filepath = file_path(savepath, filename)
    #     if filepath is not None:
    #         plt.savefig(filepath, bbox_inches='tight')
    #     plt.show()

    def plot_shifts(self, max_fret=12, show=True, harm=[], savepath=None, filename=None):
        fret_list = np.arange(0, max_fret + 1)
        shifts = self._freq_shifts(fret_list[1:])
        if harm:
            zero_strings = harm[0]
            zero_frets = harm[1]
            for s, n in zip(zero_strings, zero_frets):
                shifts[s-1] -= shifts[s-1][n]
        rms = np.sqrt(np.mean(shifts**2))
        names = self._strings.get_string_names()

        plt.figure(figsize=(8.0,6.0))
        for string in self._string_list:
            plt.plot(fret_list, shifts[string-1], label='{}'.format(names[string-1]))
        plt.xlabel('FRET', fontdict=font)
        plt.ylabel('SHIFT (cents)', fontdict=font)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', labelsize=labelsize)
        plt.xlim(fret_list[0],fret_list[-1])
        plt.ylim(-4,10)
        plt.legend(loc='upper right', fontsize=labelsize)
        
        if np.all(np.abs(self._ds - self._ds[0]) < 1.0e-06):
            ds = self._ds[0]
            template_ds = '$\Delta S = {}~\mathrm{{mm}}$'
        else:
            ds = np.round(np.mean(self._ds), 2)
            template_ds = '$\Delta S = {}~\mathrm{{mm~(mean)}}$'
        if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
            dn = self._dn[0]
            template_dn = '$\Delta N = {}~\mathrm{{mm}}$'
        else:
            dn = np.round(np.mean(self._dn), 2)
            template_dn = '$\Delta N = {}~\mathrm{{mm~(mean)}}$'
        template = '{}\n' + template_ds + '\n' + template_dn + '\n' + 'Shift~(rms):~{}~cents'
        plt.text(0.35, 5.3, template.format(self._name, ds, dn, np.round(rms, 2)),
                 fontdict=font, bbox=bbox)
        plt.grid(True)
        
        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            plt.savefig(filepath, bbox_inches='tight')
            print("Saved {0}\n".format(filepath))
        if show:
            plt.show()
        else:
            plt.close()

    # def plot_shifts_old(self, max_fret=12, show=True, harm=[], savepath=None, filename=None):
    #     fret_list = np.arange(0, max_fret + 1)
    #     shifts = self.freq_shifts(fret_list[1:])
    #     if harm:
    #         zero_strings = harm[0]
    #         zero_frets = harm[1]
    #         for s, n in zip(zero_strings, zero_frets):
    #             shifts[s-1] -= shifts[s-1][n]
    #     rms = np.sqrt(np.mean(shifts**2))
    #     names = self._strings.get_string_names()

    #     # labelsize = 18
    #     # fontsize = 24
    #     # font = {'family' : 'serif',
    #     #         'color'  : 'black',
    #     #         'weight' : 'normal',
    #     #         'size'   : fontsize,
    #     #         }
    #     # plt.rc('text', usetex=True)
    #     # plt.rc('font', family='serif')
        
    #     plt.figure(figsize=(8.0,6.0))
    #     for string in self._string_list:
    #         plt.plot(fret_list, shifts[string-1], label='{}'.format(names[string-1]))
    #     plt.xlabel('FRET', fontdict=font)
    #     plt.ylabel('SHIFT (cents)', fontdict=font)
    #     plt.tick_params(axis='x', labelsize=labelsize)
    #     plt.tick_params(axis='y', labelsize=labelsize)
    #     plt.xlim(fret_list[0],fret_list[-1])
    #     plt.ylim(-4,10)
    #     plt.legend(loc='upper right', fontsize=labelsize)
        
    #     if np.all(np.abs(self._ds - self._ds[0]) < 1.0e-06):
    #         ds = self._ds[0]
    #         template_ds = '$\Delta S = {}~\mathrm{{mm}}$'
    #     else:
    #         ds = np.round(np.mean(self._ds), 2)
    #         template_ds = '$\Delta S = {}~\mathrm{{mm~(mean)}}$'
    #     if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
    #         dn = self._dn[0]
    #         template_dn = '$\Delta N = {}~\mathrm{{mm}}$'
    #     else:
    #         dn = np.round(np.mean(self._dn), 2)
    #         template_dn = '$\Delta N = {}~\mathrm{{mm~(mean)}}$'
    #     template = '{}\n' + template_ds + '\n' + template_dn + '\n' + 'Shift~(rms):~{}~cents'
    #     plt.text(0.35, 5.3, template.format(self._name, ds, dn, np.round(rms, 2)),
    #              fontdict=font, bbox=bbox)
    #     plt.grid(True)
        
    #     filepath = file_path(savepath, filename)
    #     if filepath is None:
    #         pass
    #     else:
    #         plt.savefig(filepath, bbox_inches='tight')
    #         print("Saved {0}\n".format(filepath))
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()

    # def plot_harm(self, zero_strings, zero_frets, savepath=None, filename=None):
    #     fret_list = np.arange(0, 13)
    #     shifts = self.freq_shifts(fret_list[1:])
    #     for s, n in zip(zero_strings, zero_frets):
    #         shifts[s-1] -= shifts[s-1][n]
    #     rms = np.sqrt(np.mean(shifts**2))
    #     names = self._strings.get_string_names()

    #     # rms = np.sqrt(np.mean(shifts[:,1:]**2, axis=1))
    #     # rmsstr = 'RMS Frequency Errors (Harmonic Tuning): ['
    #     # template = '{:.{prec}}, '
    #     # for m in np.arange(rms.size):
    #     #     rmsstr += template.format(rms[m], prec = 2)
    #     # print(rmsstr[0:-2] + ']\n')

    #     # labelsize = 18
    #     # fontsize = 24
    #     # font = {'family' : 'serif',
    #     #         'color'  : 'black',
    #     #         'weight' : 'normal',
    #     #         'size'   : fontsize,
    #     #         }
    #     # plt.rc('text', usetex=True)
    #     # plt.rc('font', family='serif')
        
    #     plt.figure(figsize=(8.0,6.0))
        
    #     for string in self._string_list:
    #         plt.plot(fret_list, shifts[string-1], label='{}'.format(names[string-1]))
    #     plt.xlabel('FRET', fontdict=font)
    #     plt.ylabel('SHIFT (cents)', fontdict=font)
    #     plt.tick_params(axis='x', labelsize=labelsize)
    #     plt.tick_params(axis='y', labelsize=labelsize)
    #     plt.xlim(fret_list[0],fret_list[-1])
    #     plt.ylim(-4,10)
    #     plt.legend(loc='upper right', fontsize=labelsize)
        
    #     if np.all(np.abs(self._ds - self._ds[0]) < 1.0e-06):
    #         ds = self._ds[0]
    #         template_ds = '$\Delta S = {}~\mathrm{{mm}}$'
    #     else:
    #         ds = np.round(np.mean(self._ds), 2)
    #         template_ds = '$\Delta S = {}~\mathrm{{mm~(mean)}}$'
    #     if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
    #         dn = self._dn[0]
    #         template_dn = '$\Delta N = {}~\mathrm{{mm}}$'
    #     else:
    #         dn = np.round(np.mean(self._dn), 2)
    #         template_dn = '$\Delta N = {}~\mathrm{{mm~(mean)}}$'
    #     template = '{}\n' + template_ds + '\n' + template_dn + '\n' + 'Shift~(rms):~{}~cents'
    #     plt.text(0.35, 5.3, template.format(self._name, ds, dn, np.round(rms, 2)),
    #              fontdict=font, bbox=bbox)
    #     plt.grid(True)
        
    #     filepath = file_path(savepath, filename)
    #     if filepath is not None:
    #         plt.savefig(filepath, bbox_inches='tight')
    #     plt.show()
    
    def save_setbacks_table(self, max_fret:int=12, show:bool=True, savepath=None, filename=None):
        fret_list = np.arange(1, max_fret)
        dnu = self._freq_shifts(fret_list)
        rms = np.sqrt(np.mean(dnu[:,1:]**2, axis=1))

        df = pd.DataFrame({'String': self._strings.get_string_names(),
                           '$\Delta S$ (mm)': self._ds.tolist(),
                           '$\Delta N$ (mm)': self._dn.tolist(),
                           '$\overline{\Delta \\nu}_\\text{rms}$ (cents)': rms.tolist()})

        styler = df.style.format(precision=2).hide()
        table_str = styler.to_latex(column_format='cccccc', hrules=True)

        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            print("Saved {0}\n".format(filepath))
            print(table_str,  file=open(filepath, 'w'))        

        if show:
            styler.set_properties(**{'text-align': 'center'})
            display(styler)
