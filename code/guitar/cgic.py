import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from IPython.display import display
from scipy.optimize import curve_fit, minimize
import sys
import os

# Define a reasonable set of matplotlib parameters compatible
#  with Jupyter notebooks
linewidth = 2.0
labelsize = 18
fontsize = 24
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : fontsize,
        }
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


def figdisp(fig, show, savepath, filename): # dispatch
        filepath = file_path(savepath, filename)
        if filepath is None:
            pass
        else:
            fig.savefig(filepath, bbox_inches='tight')
            print("Saved {0}\n".format(filepath))

        if show:
            plt.show()
        else:
            plt.close()


def tabdisp(df, formatter, show, savepath, filename):
    styler = df.style.format(formatter=formatter).hide()
    table_str = styler.to_latex(column_format=df.shape[1]*'c', hrules=True)

    filepath = file_path(savepath, filename)
    if filepath is None:
        pass
    else:
        print(table_str,  file=open(filepath, 'w'))        
        print("Saved {0}\n".format(filepath))

    if show:
        styler.set_properties(**{'text-align': 'center'})
        display(styler)


def classmro(myself):
    ''' A function to parse the __mro__ property of a Python class
        to display its inheritance pedigree. Ambiguous results
        occur when there's multiple inheritance.

    Parameters
    ----------
    myself : class instance
        Usually 'self' if classmro is used within a class

    Returns
    ----------
    retval : str
        A string listing the classes from which 'myself' is derived
        (including 'myself').
    '''

    class_str = myself.__class__.__name__
    for classname in myself.__class__.__mro__[1:-1]:
        class_str += " : {}".format(classname.__name__)
    return class_str


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


def setarr(x, count:int):
    if count == 0 or count is None:
        return x
    
    if isinstance(x, np.ndarray):
        assert x.shape == (count,), 'Input array has shape {}, not {}.'.format(x.shape, (count,))
        return x.astype(np.float)
    elif isinstance(x, list):
        if len(x) == 1:
            return np.array(count * x, dtype=np.float64)
        else:
            assert len(x) == count, 'Input list has length {}, not {}.'.format(len(x), count)
            return np.array(x, dtype=np.float64)
    else:
        return np.array(count * [x], dtype=np.float64)


class BaseClass(object):
    '''A general-purpose virtual base class for functions of time (or space).

    Required Private Methods
    ------------------------
    _set_key_list :
        List of keys that will be present in a parameter dictionary
        that will be required by a derived class; default = []
    '''

    def __init__(self, params:dict, count:int=0):
        '''Define the list of dictionary keys of parameters required by the
        derived class, and then set those parameters
        `'''
        self._set_specs()
        self.set_params(params, count)
        
    def _set_specs(self):
        '''Define the names (strings) of keyword parameters that must be supplied
        through __init__() and from them create a private list variable _param_list.
        For example, if the required parameters are "a" and "b", then in the
        derived class:
            def _set_param_list(self):
                self._param_list = ['a', 'b']
        Example of use by a derived class:
            obj = DerivedClass(a=1.0, b=2.0)
        '''
        self._specs = dict()
        
    def _check_params(self, params:dict):
        specs_keys = set(self._specs.keys())
        dict_keys = set(key[1:] for key in self.__dict__.keys())
        params_keys = set(params.keys())
        
        extra = sorted(params_keys - specs_keys)
        missing = sorted( (specs_keys - params_keys)
                        & (specs_keys - dict_keys) )
        
        assert not bool(extra), 'Extra keys in params dict: {}'.format(extra)
        assert not bool(missing), 'Missing keys in params dict: {}'.format(missing)

    def __str__(self):
        ''' Return a string containing the attributes of an object derived from
            the base class. Example:
                obj = DerivedClass(...)
                print(obj)
        '''
        param_str = classmro(self) + '\n'
        for key, value in self.__dict__.items():
            if key == '_key_list':
                continue
            if not isinstance(value, BaseClass):
                param_str += "\t{} : {}\n".format(key[1:], value)
            else:
                param_str += "\n\t{} : {}".format(key[1:], value)

        return param_str

    def __rich__(self):
        ''' Return a string containing the attributes of an object derived from
        the base class using rich text. Example:
                from rich import print
                obj = DerivedClass(...)
                print(obj)
        '''
        param_str = "[bold blue]{}\n".format(classmro(self))
        for key, value in self.__dict__.items():
            if key == '_key_list':
                continue
            if isinstance(value, float):
                param_str += "\t[green]{} : {}\n".format(key[1:], value)
            elif isinstance(value, int):
                param_str += "\t[red]{} : {}\n".format(key[1:], value)
            elif not isinstance(value, BaseClass):
                param_str += "\t{} : {}\n".format(key[1:], value)
            else:
                param_str += "\n\t[cyan]{} : {}".format(key[1:], value.__rich__())

        return param_str

    def set_params(self, params:dict, count:int=0):
        '''Walk through the list of keyword parameters, and for each one
        create a private variable with a name that is the input parameter name
        preceded by an underscore (i.e., 'varname' becomes '_varname')
        '''
        self._check_params(params)
        for key, value in params.items():
            if self._specs[key]:
                self.__dict__['_' + key] = setarr(value, count)
            else:
                self.__dict__['_' + key] = value


class GuitarString(object):
    '''Collect parameters and compute properties of guitar strings
       Estimate the frequency shift (due to frequency-pulling and dispersion)
       and the round-trip time delay (due to dispersion) of each mode q.
    
    Public Methods
    --------------
    set_scale_length : 
        Set the scale length of the guitar (which rescales the open string tension)
    fit_r :
        Fit data on frequency change with length to determine R
    get_specs : pandas Series
        Return a pandas Series with the string's specifications in MKS units
    get_props : pandas Series
        Return a pandas Series with the string's properties in MKS units
    '''

    def __init__(self, specs, props, scale_length):
        '''Initialize a GuitarString object.

        Parameters
        ----------
        specs : pandas Series
            A pandas Series with the following elements:
                'string' : str
                    A python string containing the name of the guitar string
                'note': str
                    A python string labeling the fundamental frequency of the
                    open string using scientific notation, such as 'A_4', 'Ab_4',
                    or 'A#_4'
                'radius' : float
                    The radius of the string in mm
                'density': float
                    The linear mass density of the string in mg/mm
                'tension': float
                    The nominal tension of the guitar string in newtons
                'scale': float
                    The scale length of the guitar (2x the distance measured
                    from the inside edge of the nut to the center of the twelfth
                    fret) in mm; this is needed as a reference for the tension
        props : pandas Series
            If not 'None', a pandas Series with the following elements:
                'string' : str
                    A python string containing the name of the guitar string
                'r' : float
                    The (dimensionless) R parameter of the string
                'sigma' : float
                    The (dimensionless) covariant standard deviation of R
                'kappa' : float
                    The (dimensionless) string constant (2 * R + 1)
                'b_0' : float
                    The (dimensionless) bending stiffness of the open string
                'e_eff' : float
                    The effective elastic modulus of the string material in GPa
        scale_length : float
            The scale length of the guitar in mm, which determines the open-string
            tension
        '''
        self._specs = specs
        self._props = props
        
        self._freq = self._frequency(self._specs.note)
        self.set_scale_length(scale_length)

    def _frequency(self, note_str):
        ''' Compute the frequency of a musical note
        
        Parameters
        ----------
        note_str : str
            A python string labeling a musical note using scientific notation,
            such as 'A_4', 'Ab_4',  or 'A#_4'
        
        Returns
        --------
        float
            The frequency in Hertz of the musical note
        '''
        notes = dict([('Ab', 49), ('A', 48), ('A#', 47), ('Bb', 47), ('B', 46), ('B#', 57), ('Cb', 46), ('C', 57), ('C#', 56),
                  ('Db', 56), ('D', 55), ('D#', 54), ('Eb', 54), ('E', 53), ('E#', 52), ('Fb', 53), ('F', 52), ('F#', 51),
                  ('Gb', 51), ('G', 50), ('G#', 49)])

        note = note_str.split('_')
        return 440.0 * 2**( int(note[1], 10) - notes[note[0]]/12.0 )

    def _set_props(self, r, dr, scale_length):
        kappa = 2 * r + 1
        b_0 = np.sqrt(kappa) * self._specs.radius / ( 2 * scale_length )
        e_eff = 1.0e-09 * (self._specs.tension / (np.pi * (self._specs.radius/1000)**2)) * kappa
        d =  {'string' : self._specs.string, 'r' : r, 'sigma' : dr, 'kappa' : kappa, 'b_0' : b_0, 'e_eff' : e_eff}
        
        self._props = pd.Series(data=d, name=self._specs.name)
    
    def set_scale_length(self, scale_length):
        ''' Compute the tension of an open string for a guitar with a
            particular scale length
            
        Parameters
        ----------
        scale_length : float
            The scale length of the guitar in mm
        '''
        mu = self._specs.density / 1000    # Convert mg/mm to kg/m
        x0 = scale_length / 1000      # Convert mm to m
        self._specs.scale = scale_length
        self._specs.tension = mu * (2 * x0 * self._freq)**2
        
    def fit_r(self, scale_length, scale_dx, dx, df, ddf):
        def func(x, intercept, slope):
            return intercept + slope * x

        param, param_cov = curve_fit(func, scale_dx*dx, df, sigma=ddf)
        fit = func(scale_dx*dx, *param)
        
        dfdx = param[1]
        ddfdx = np.sqrt(param_cov[1][1])
        
        r = (scale_length / self._freq) * dfdx
        dr = (scale_length / self._freq) * ddfdx
        
        self._set_props(r, dr, scale_length)

        return fit

    def get_specs(self):
        ''' Return a pandas Series with the string's specifications in MKS units
        
            See __init()__ for a description of the elements of this Series
        '''
        return self._specs

    def get_props(self):
        ''' Return a pandas Series with the string's specifications in MKS units
        
            See __init()__ for a description of the elements of this Series
        '''
        return self._props


class GuitarStrings(object):
    def __init__(self, name, path_specs, path_props, sheet_name=0, scale_length=650.0, units='IPS'):
        self._name = name

        df_specs = pd.read_excel(path_specs, sheet_name=sheet_name,
                                 dtype={'string' : str, 'note': str, 'diameter' : np.float64,
                                        'density': np.float64, 'tension': np.float64, 'scale': np.float64})
        df_specs.diameter /= 2
        df_specs.rename(columns={"diameter" : "radius"}, inplace=True)
        if units == 'IPS':
            in_to_mm = 25.4
            lb_to_mg = 453592.37
            lb_to_nt = 4.4482216153 # 9.81 / 2.204

            df_specs.radius *= in_to_mm
            df_specs.density *= (lb_to_mg/in_to_mm)
            df_specs.tension *= lb_to_nt
            df_specs.scale *= in_to_mm
        indices, rows_specs = zip(*df_specs.iterrows())

        if path_props is None:
            rows_props = (None,) * df_specs.shape[0]
        else:
            df_props = pd.read_excel(path_props, sheet_name=sheet_name,
                                     dtype={'string' : str, 'R': np.float64, 'sigma' : np.float64, 'kappa': np.float64,
                                            'B_0': np.float64, 'E': np.float64})
            assert df_specs['string'].equals(df_props['string']), 'Specification string names and Property string names do not match.'
            indices, rows_props = zip(*df_props.iterrows())
        
        self._strings = []
        for row_specs, row_props in zip(rows_specs, rows_props):
            string = GuitarString(row_specs, row_props, scale_length)
            self._strings.append(string)
        
        self._build_specs_frame()
        if path_props is None:
            self._props = None
        else:
            self._build_props_frame()

    def __str__(self):
        '''Return a string displaying the attributes of a GuitarStrings object.
    
        Example
        -------
        strings = GuitarStrings(name, path_specs, path_props)
        
        print(strings)
        '''
        
        df = self._specs.copy()
        df.drop(columns=["scale"], inplace=True)
        df.rename(columns={"string" : "String", "note" : "Note", "radius" : "Radius (mm)",
                           "density" : "Density (mg/mm)", "tension" : "Tension (N)"},
                  inplace=True)
        formatters = {'Radius (mm)': '{:.3f}'.format,
                      'Density (mg/mm)': '{:.3f}'.format,
                      'Tension (N)': '{:.1f}'.format}

        str_specs = df.to_string(index=False, justify='center', formatters=formatters)
        
        if self._props is None:
            return str_specs
        else:
            df = self._props.copy()
            df.rename(columns={"string" : "String", "r" : "R", "sigma" : "R std",
                            "kappa" : "kappa ", "b_0" : "B_0", "e_eff" : "E_eff (GPa)"},
                    inplace=True)
            formatters = {'R': '{:.1f}'.format,
                          'R std': '{:.1f}'.format,
                          'kappa ': '{:.1f}'.format,
                          'B_0': '{:.5f}'.format,
                          'E_eff (GPa)': '{:.2f}'.format}

            str_props = df.to_string(index=False, justify='center', formatters=formatters)
            return str_specs + '\n' + str_props

    def _check_string_names(self, data):
        specs_names = sorted(self._specs.copy()['string'].to_list())
        data_names = sorted(list(data.columns)[1:])
        assert len(specs_names) == len(data_names), 'This string set has {} strings, not {}.'.format(len(specs_names), len(data_names))
        assert specs_names == data_names, 'The input string names should be {}, not {}.'.format(specs_names, data_names)
            
    def _build_specs_frame(self):
        series_specs = []
        for string in self._strings:
            series_specs.append(string._specs)
        
        self._specs = pd.DataFrame(series_specs)
        
    def _build_props_frame(self):
        series_props = []
        for string in self._strings:
            series_props.append(string._props)
        
        self._props = pd.DataFrame(series_props)
        
    def set_scale_length(self, scale_length):
        for string in self._strings:
            string.set_scale_length(scale_length)

    def get_count(self):
        return len(self._strings)
    
    def get_specs(self):
        return self._specs
    
    def get_props(self):
        return self._props

    def fit_r(self, data_path, sheet_name=0, sigma_name=None, scale_length=650.0, scale_dx=2**(1/12),
              show=True, save_path=None, file_name=None, markersize=12.5):
        data = pd.read_excel(data_path, sheet_name=sheet_name)
        self._check_string_names(data)
        if sigma_name is None:
            sigma = None
        else:
            sigma = pd.read_excel(data_path, sheet_name=sigma_name)

        dx = data['dx'].to_numpy()
        fit_dict = {}
        for string in self._strings:
            name = string.get_specs()['string']
            if sigma is None:
                ddf = None
            else:
                ddf = sigma[name].to_numpy()
            fit = string.fit_r(scale_length, scale_dx, dx, data[name].to_numpy() -  data[name].to_numpy()[0], ddf=ddf)
            fit_dict[name] = fit
        self._build_props_frame()
        
        self.plot_fit(fit_dict, data, sigma, show, save_path, file_name, markersize)
    
    def plot_fit(self, fit_dict, data, sigma, show, savepath, filename, markersize):
        dx = np.array(data[[list(data.columns)[0]]].values.T[0])
        
        fig, ax = plt.subplots(figsize=(8.0,6.0))
        if sigma is None:
            for string in self._strings:
                name = string.get_specs()['string']
                ax.plot(dx, data[name].values -  data[name].values[0], '.', markersize=markersize)
                ax.plot(dx, fit_dict[name], color=plt.gca().lines[-1].get_color(), linewidth=linewidth, label='{}'.format(name))
        else:
            for string in self._strings:
                name = string.get_specs()['string']
                ax.errorbar(dx, data[name].values -  data[name].values[0], fmt='.', markersize=markersize,
                             yerr=sigma[name], capsize=5, capthick=1)
                ax.plot(dx, fit_dict[name], color=plt.gca().lines[-1].get_color(), linewidth=linewidth, label='{}'.format(name))

        ax.set_xlabel(r'$\Delta x$~(mm)', fontdict=font)
        ax.set_ylabel(r'$\Delta f$~(Hz)', fontdict=font)
        ax.set_xlim(dx[0], dx[-1])
        ax.set_ylim(0, get_ylim()[1])
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.legend(loc='upper left', fontsize=labelsize)
        ax.grid(visible=True)

        figdisp(fig, show, savepath, filename)
 
    def save_specs_table(self, show=True, savepath=None, filename=None):
        df = self._specs.copy()
        df.drop(columns=["scale"], inplace=True)
        notes = []
        for note in df.note:
            note_str = note.split('_')
            notes.append(note_str[0] + '$_{' + note_str[1] + '}$')
        df.replace(df.note.tolist(), notes, inplace=True)

        df.rename(columns={"string" : "String", "note" : "Note", "radius" : "$\\rho$ (mm)",
                           "density" : "$\mu$ (mg/mm)", "tension" : "$T_0$ (N)"},
                  inplace=True)

        formatter = {'$\\rho$ (mm)': '{:.3f}',
                     '$\mu$ (mg/mm)': '{:.3f}',
                     '$T_0$ (N)': '{:.1f}'}

        tabdisp(df, formatter, show, savepath, filename)      
 
    def save_props_table(self, show=True, savepath=None, filename=None):
        df = self._props.copy()
        df.rename(columns={"string" : "String", "r" : "$R$", "sigma" : "$\sigma$",
                           "kappa" : "$\kappa$", "b_0" : "$B_0$", "e_eff" : "$E_\mathrm{eff}$ (GPa)"},
                  inplace=True),

        formatter = {'$R$': '{:.1f}',
                     '$\sigma$': '{:.1f}',
                     '$\kappa$': '{:.1f}',
                     '$B_0$': '{:.5f}',
                     '$E_\mathrm{eff}$ (GPa)': '{:.2f}'}
 
        tabdisp(df, formatter, show, savepath, filename)      

    def save_props_excel(self, filepath, sheet_name):
        df = self._props.copy()

        if os.path.isfile(filepath):
            with pd.ExcelWriter(filepath,  mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        else:
            df.to_excel(filepath, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        print("\nSaved {} : {}\n".format(filepath, sheet_name))
        

class Guitar(BaseClass):
    def __init__(self, params, string_count, strings):
        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self._strings = strings
    
        self._set_specs()
        self.set_params(params, string_count)
        self._strings = strings

    def _set_specs(self):
        self._specs = { 'name' : False,
                        'x0' : False,
                        'ds' : True,
                        'dn' : True,
                        'b' : True,
                        'c' : True,
                        'd' : False,
                        'rgx' : True }

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
        if np.all(np.abs(self._dn - self._dn[0]) < 1.0e-06):
            retstr += 'Nut Setback: ' + '{:.2f} mm\n'.format(self._dn[0])
        else:
            retstr += 'Nut Setback: ' + np.array2string(self._dn[0], precision=2, floatmode='fixed', separator=', ') + ' mm\n'
        if np.all(np.abs(self._b - self._b[0]) < 1.0e-06):
            retstr += 'b: ' + '{:.2f} mm\n'.format(self._b[0])
        else:
            retstr += 'b: ' + np.array2string(self._b, precision=2, floatmode='fixed', separator=', ') + ' mm\n'
        if np.all(np.abs(self._c - self._c[0]) < 1.0e-06):
            retstr += 'c: ' + '{:.2f} mm\n'.format(self._c[0])
        else:
            retstr += 'c: ' + np.array2string(self._c, precision=2, floatmode='fixed', separator=', ') + ' mm\n'
        retstr += 'd: ' + '{:.1f} mm\n'.format(self._d)
        if np.all(np.abs(self._rgx - self._rgx[0]) < 1.0e-06):
            retstr += 'Radius of Gyration Correction: ' + '{:.2f}\n'.format(self._rgx[0])
        else:
            retstr += 'Radius of Gyration Correction: ' + np.array2string(self._rgx, precision=2, floatmode='fixed', separator=', ') + '\n'
        retstr += self._strings.__str__() + "\n"

        return retstr

    def _tile_strings(self, x):
        return np.tile(x, (self._strings.get_count(), 1))
    
    def _tile_frets(self, x, fret_list):
        return np.tile(x, (fret_list.size, 1)).T
    
    def _rms(self, dnu):
        rms = np.sqrt(np.mean(dnu**2))
        return rms

    def _gamma(self, n):
        return 2.0**(n/12.0)

    def _l0(self, ds, dn, x0, c):
        length = np.sqrt( (x0 + ds + dn)**2 + c**2 )
        return length

    def _l(self, ds, x0, b, c, n):
        length = np.sqrt( (x0/self._gamma(n) + ds)**2 + (b + c)**2 )
        return length

    def _lp(self, ds, dn, x0, b, c, d, n):
        xn = x0 / self._gamma(n)
        ln = self._l(ds, x0, b, c, n)
        length = np.sqrt( (x0 - xn + dn - d)**2 + (b + (b + c) * d / (xn + ds))**2 )
        length += ( ln / (xn + ds)  ) * d
        return length

    def _lmc(self, ds, dn, x0, b, c, d, n):
        length = self._l(ds, x0, b, c, n) + self._lp(ds, dn, x0, b, c, d, n)
        return length
    
    def _q(self, ds, dn, x0, b, c, d, n):
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
        kappa = self._tile_frets(self._strings.get_props().kappa.to_numpy(), fret_list)
        b_0 = self._tile_frets(self._strings.get_props().b_0.to_numpy() * self._rgx , fret_list)
        n = self._tile_strings(fret_list)
        
        shifts = self._tfe(ds, dn, x0, b, c, d, kappa, b_0, n)
        
        open_strings = np.array(self._strings.get_count() * [0]).reshape(-1,1)                       
        
        return np.hstack((open_strings, shifts))
    
    # def set_vars(self, **kwargs):
    #     count = self._strings.get_count()
        
    #     x0 = kwargs.get('x0')
    #     if x0 is None:
    #         pass
    #     else:
    #         self._x0 = x0
        
    #     ds = kwargs.get('ds')
    #     if ds is None:
    #         pass
    #     else:
    #         self._ds = setarr(ds, count)

    #     dn = kwargs.get('dn')
    #     if dn is None:
    #         pass
    #     else:
    #         self._dn = setarr(dn, count)

    #     b = kwargs.get('b')
    #     if b is None:
    #         pass
    #     else:
    #         self._b = setarr(b, count)

    #     c = kwargs.get('c')
    #     if c is None:
    #         pass
    #     else:
    #         self._c = setarr(c, count)

    #     d = kwargs.get('d')
    #     if d is None:
    #         pass
    #     else:
    #         self._d = d

    #     rgx = kwargs.get('rgx')
    #     if rgx is None:
    #         pass
    #     else:
    #         self._rgx = setarr(rgx, count)

    def approximate(self):
        x_0 = self._x0
        b = self._b
        c = self._c
        d = self._d
        kappa = self._strings.get_props().kappa.to_numpy()
        b_0 = self._strings.get_props().b_0.to_numpy() * self._rgx

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
        kappa = self._tile_frets(self._strings.get_props().kappa.to_numpy(), fret_list)
        b_0 = self._tile_frets(self._strings.get_props().b_0.to_numpy() * self._rgx, fret_list)
        n = self._tile_strings(fret_list)
        ds = np.zeros(n.shape)
        dn = np.zeros(n.shape)
        
        mde = self._mde(ds, dn, x0, b, c, d, n) - (1200/np.log(2)) * ( self._gamma(n)**2 * (b + c)**2 - c**2 ) / (2 * x0**2)
        tse = self._tse(ds, dn, x0, b, c, d, kappa, n)
        bse = self._bse(b_0, n)
        
        z_n = (np.log(2.0) / 1200.0) * ( mde + tse + bse )
        
        g_0 = max_fret
        g_1 = sigma_k(fret_list, 1)
        g_2 = sigma_k(fret_list, 2)
        
        zbar_0 = np.sum(z_n, axis=1)
        zbar_1 = np.sum((self._gamma(fret_list) - 1) * z_n, axis=1)
        
        det = g_0 * g_2 - g_1**2
        ds =  ( g_0 * zbar_1 - g_1 * zbar_0 ) * self._x0 / det
        dn = -( g_2 * zbar_0 - g_1 * zbar_1 ) * self._x0 / det

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
        kappa = self._strings.get_props().kappa.to_numpy()
        b_0 = self._strings.get_props().b_0.to_numpy() * self._rgx

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

    def plot_shifts(self, max_fret=12, show=True, harm=[], savepath=None, filename=None, markersize=9.0, alpha=1.0):
        fret_list = np.arange(0, max_fret + 1)
        shifts = self._freq_shifts(fret_list[1:])
        if harm:
            zero_strings = harm[0]
            zero_frets = harm[1]
            for s, n in zip(zero_strings, zero_frets):
                shifts[s-1] -= shifts[s-1][n]
        rms = np.sqrt(np.mean(shifts**2))
        names = self._strings.get_specs().string.tolist()

        fig, ax = plt.subplots(figsize=(8.0,6.0))
        for index in np.arange(self._strings.get_count()):
            ax.plot(fret_list, shifts[index], '.', markersize=markersize)
            ax.plot(fret_list, shifts[index], color=plt.gca().lines[-1].get_color(), alpha=alpha, linewidth=linewidth, label='{}'.format(names[index]))

        ax.set_xlabel('FRET', fontdict=font)
        ax.set_ylabel('SHIFT (cents)', fontdict=font)
        ax.set_xlim(fret_list[0],fret_list[-1])
        ax.set_ylim(-2,8)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.legend(loc='upper right', fontsize=labelsize)
        ax.grid(visible=True)
        
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
        annotation = template.format(self._name, ds, dn, np.round(rms, 2))
        ob = offsetbox.AnchoredText(annotation, loc='upper left', pad=0, borderpad=0.65, prop=dict(size=fontsize))
        ob.patch.set(boxstyle='round', edgecolor='#D7D7D7', facecolor='white', alpha=0.75)
        ax.add_artist(ob)
        
        figdisp(fig, show, savepath, filename)
    
    def save_setbacks_table(self, max_fret:int=12, show:bool=True, savepath=None, filename=None):
        fret_list = np.arange(1, max_fret)
        dnu = self._freq_shifts(fret_list)
        rms = np.sqrt(np.mean(dnu[:,1:]**2, axis=1))

        df = pd.DataFrame({'String': self._strings.get_specs().string.tolist(),
                           '$\Delta S$ (mm)': self._ds.tolist(),
                           '$\Delta N$ (mm)': self._dn.tolist(),
                           '$\overline{\Delta \\nu}_\\text{rms}$ (cents)': rms.tolist()})

        formatter = {'$\Delta S$ (mm)': '{:.2f}',
                     '$\Delta N$ (mm)': '{:.2f}',
                     '$\overline{\Delta \\nu}_\\text{rms}$ (cents)': '{:.3f}'}

        tabdisp(df, formatter, show, savepath, filename)      
