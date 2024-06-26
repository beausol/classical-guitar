import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from scipy.optimize import curve_fit, minimize
import sys
import os
from guitar.utils import *

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
        '''
        Initialize a GuitarString object.

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

    def _frequency(self, note_str:str):
        '''
        Compute the frequency of a musical note
        
        Parameters
        ----------
        note_str : str
            A python string labeling a musical note using scientific notation,
            such as 'A_4', 'Ab_4',  or 'A#_4'
        
        Returns
        --------
        retval : float
            The frequency in Hertz of the musical note
        '''
        notes = dict([('Ab', 49), ('A', 48), ('A#', 47), ('Bb', 47), ('B', 46), ('B#', 57), ('Cb', 46), ('C', 57), ('C#', 56),
                  ('Db', 56), ('D', 55), ('D#', 54), ('Eb', 54), ('E', 53), ('E#', 52), ('Fb', 53), ('F', 52), ('F#', 51),
                  ('Gb', 51), ('G', 50), ('G#', 49)])

        note = note_str.split('_')
        return 440.0 * 2**( int(note[1], 10) - notes[note[0]]/12.0 )

    def _set_props(self, r:float, dr:float):
        '''
        Create a pandas Series containing the properties of the string
        
        Parameters
        ----------
        r : float
            The R parameter of the string determined by a fit to data
        dr : float
            The standard deviation of R
        
        See __init()__ for a description of the elements of this Series
        '''
        kappa = 2 * r + 1
        b_0 = np.sqrt(kappa) * self._specs.radius / ( 2 * self._specs.scale )
        e_eff = 1.0e-09 * (self._specs.tension / (np.pi * (self._specs.radius/1000)**2)) * kappa
        d =  {'string' : self._specs.string, 'r' : r, 'sigma' : dr, 'kappa' : kappa, 'b_0' : b_0, 'e_eff' : e_eff}
        
        self._props = pd.Series(data=d, name=self._specs.name)
    
    def set_scale_length(self, scale_length):
        '''
        Compute the tension of an open string for a guitar with a
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

    def fit_r(self, scale_dx:float, dx:float, df:float, sigma:float):
        '''
        Find the R parameter of a string by linear least-squares fit
        of frequency shift data as a function of differential stretch
        
        Parameters
        ----------
        scale_dx : float
            Scale the values of dx so that they represent the change in
            the length of the open string; for example, if the measurements
            of dx are made near the first fret, scale_dx = 2**(1/12)
        dx : numpy.ndarray
            The incremental incease in the open string length
        df : numpy.ndarray
            The incremental increase in the open string vibration frequency
        sigma : numpy.ndarray
            The uncertainty in df; None if not measured/available
        
        Returns
        -------
        retval : numpy:ndarray
            Line of best fit (computed at dx)
        '''
        def func(x, intercept, slope):
            return intercept + slope * x

        param, param_cov = curve_fit(func, scale_dx*dx, df, sigma=sigma)
        fit = func(scale_dx*dx, *param)
        
        dfdx = param[1]
        ddfdx = np.sqrt(param_cov[1][1])
        
        r = (self._specs.scale / self._freq) * dfdx
        dr = (self._specs.scale / self._freq) * ddfdx
        
        self._set_props(r, dr)

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
        '''
        Initialize a GuitarStrings object
        
        Parameters
        ----------
        name : str
            The name of the string set
        path_specs : str
            A full path to the Excel file containing the string set specifications
        path_props : str
            A full path to the Excel file containing the string set properties
        sheet_name : str
            If not zero, the sheet in the specs and props files containing the
            specifications and properties for each string in a string set;
            else, if zero (the default) read the first sheet
        scale_length : float
            The scale length of the guitar (in mm); needed to set the tension
            of each string corresponding to each string's open frequency
        units : str
            If 'IPS' (the default), then the units of the parameters listed in
            the string specifications file follow the British Imperial measurement
            system, and are converted to metric units (MKS); otherwise, the
            units must follow the conventions listed in GuitarString.__init__().
        '''
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
        '''
        Check that the string names in the data file match those
        in the string set specifications
        '''
        specs_names = sorted(self._specs.copy()['string'].to_list())
        data_names = sorted(list(data.columns)[1:])
        assert len(specs_names) == len(data_names), 'This string set has {} strings, not {}.'.format(len(specs_names), len(data_names))
        assert specs_names == data_names, 'The input string names should be {}, not {}.'.format(specs_names, data_names)
            
    def _build_specs_frame(self):
        '''
        Build a pandas DataFrame containing the string set
        specifications
        '''
        series_specs = []
        for string in self._strings:
            series_specs.append(string._specs)
        
        self._specs = pd.DataFrame(series_specs)
        
    def _build_props_frame(self):
        '''
        Build a pandas DataFrame containing the string set
        properties
        '''
        series_props = []
        for string in self._strings:
            series_props.append(string._props)
        
        self._props = pd.DataFrame(series_props)
        
    def set_scale_length(self, scale_length:float):
        '''
        Set the tensions for each string in the set for a
        particular guitar scale length
        
        Parameters
        ----------
        scale_length : float
            The scale length (in mm) of the guitar
        '''
        for string in self._strings:
            string.set_scale_length(scale_length)

    def get_count(self):
        '''
        Return the number of strings in the string set
        
        Returns
        -------
        retval : int
            The number of strings in the string set
        '''
        return len(self._strings)
    
    def get_specs(self):
        '''
        Return a pandas DataFrame containing the specifications
        of the strings in the set
        
        Returns
        -------
        retval : pandas.DataFrame
            A pandas DataFrame containing the specifications of
            the strings in the set
        '''
        return self._specs
    
    def get_props(self):
        '''
        Return a pandas DataFrame containing the properties of
        the strings in the set
        
        Returns
        -------
        retval : pandas.DataFrame
            A pandas DataFrame containing the properties of
            the strings in the set
        '''
        return self._props

    def fit_r(self, data_path, sheet_name=0, sigma_name=None, scale_dx=2**(1/12),
              show=True, save_path=None, file_name=None, markersize=12.5):
        '''
        Find the R parameter of each string in a set by linear least-squares
        fit of frequency shift data as a function of differential stretch
        
        Parameters
        ----------
        data_path : str
            Full path to the Excel data file containing the strings data
        sheet_name : str
            If not zero, the sheet in the data file containing the frequency
            shift data for each string in a string set; else, if zero (the
            default) read the first sheet
        sigma_name : str
            Sheet name for measurement uncertainties for a string set; if
            None (the default), the fits are performed without uncertainties
        scale_dx : float
            Scale the values of the measured displacements so that they
            represent the change in the length of the open strings; for
            example, if the measurements are made near the first fret,
            scale_dx = 2**(1/12) (the default)
        show : bool
            If True (the default), show a plot of the results of the fits
        savepath : str
            A valid path to a directory / folder where the figure will be
            saved, or None (the default); if None, the figure is not saved.
        filename : str
            A valid file name (including an extension where needed) that
            will contain the saved figure, or None (the default); if None,
            the figure is not saved.
        markersize : float
            The size of the markers representing the data points in the plot
            of the fit results; default = 12.5
        '''
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
            fit = string.fit_r(scale_dx, dx, data[name].to_numpy() -  data[name].to_numpy()[0], ddf)
            fit_dict[name] = fit
        self._build_props_frame()
        
        self.plot_fit(fit_dict, data, sigma, show, save_path, file_name, markersize)
    
    def plot_fit(self, fit_dict, data, sigma, show, savepath, filename, markersize):
        '''
        Plot the fit results 
        
        Parameters
        ----------
        fit_dict : dict
            A dictionary with keys that are the names of the strings in the set and
            values that are the lines of best fit
        data : pandas.DataFrame
            A pandas dataframe containing the displacement and frequency shift data
            for each string
        sigma : pandas.DataFrame
            A pandas dataframe containing the standard deviations of the frequency
            data points; if None, then no error bars are included in the plots
        show : bool
            If True (the default), show the plot of the results of the fits
        savepath : str
            A valid path to a directory / folder where the figure will be
            saved, or None (the default); if None, the figure is not saved.
        filename : str
            A valid file name (including an extension where needed) that
            will contain the saved figure, or None (the default); if None,
            the figure is not saved.
        markersize : float
            The size of the markers representing the data points in the plot
            of the fit results
        '''
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
        '''
        Show and/or save (in LaTeX format) a table of the specifications of each
        string in a set
        
        Parameters
        ----------
        show : bool
            If True (the default), show the table using IPython.display.
        savepath : str
            A valid path to a directory / folder, or None; if
            None, the table is not saved.
        filename : str
            A valid file name (including an appropriate LaTeX extension),
            or None; if None, the table is not saved.
        '''
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
        '''
        Show and/or save (in LaTeX format) a table of the properties of each
        string in a set
        
        Parameters
        ----------
        show : bool
            If True (the default), show the table using IPython.display.
        savepath : str
            A valid path to a directory / folder, or None; if
            None, the table is not saved.
        filename : str
            A valid file name (including an appropriate LaTeX extension),
            or None; if None, the table is not saved.
        '''
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
        '''
        Write the properties of the string set to an Excel file
        
        Parameters
        ----------
        filepath : string
            A fully qualified path to the Excel file, which will be created
            if it doesn't exist
        sheet_name : string
            The name of the target sheet in the Excel file; it will be created
            if it doesn't exist, and overwritten if it does
        '''
        df = self._props.copy()

        if os.path.isfile(filepath):
            with pd.ExcelWriter(filepath,  mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        else:
            df.to_excel(filepath, sheet_name=sheet_name, float_format="%.{}f".format(sys.float_info.dig), index=False)
        print("\nSaved {} ({})\n".format(filepath, sheet_name))
        

class Guitar(BaseClass):
    def __init__(self, params:dict, string_count:int, strings:GuitarStrings):
        '''
        Initialize a Guitar object

        Parameters
        ----------
        params : dict
            A dict with the following keys (and values/types):
                'name' : str
                    A python string containing the name of the guitar
                'x0': float
                    The scale length of the guitar in mm
                'ds' :
                    A number, a list, or an ndarray representing the saddle
                    setbacks for each string in mm; converted to a numpy.array
                    with dtype = float64.
                'dn' :
                    A number, a list, or an ndarray representing the nut
                    setbacks for each string in mm; converted to a numpy.array
                    with dtype = float64.
                'b' : 
                    A number, a list, or an ndarray representing the height of
                    the bottom of each string above the fret board minus the
                    height of the top of the frets (in mm); converted to a
                    numpy.array with dtype = float64
                'c' :
                    A number, a list, or an ndarray representing the height of
                    the top of the saddle in mm relative to the height b;
                    converted to a numpy.array with dtype = float64
                'd': float
                    The distance in mm representing the size of the fretting finger
                'tension': float
                    The nominal tension of the guitar string in newtons
                'scale': float
                    The scale length of the guitar (2x the distance measured
                    from the inside edge of the nut to the center of the twelfth
                    fret) in mm; this is needed as a reference for the tension
        string_count : int
            The number of strings on the guitar
        strings : GuitarStrings
            The object containing the specifications and properties of the guitar
            strings; string_count == strings.get_count() must be True
        '''
        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self._strings = strings
    
        BaseClass.__init__(self, params, string_count)

    def _set_specs(self):
        '''
        Specify the names (strings) of dictionary keys that must be supplied
        through __init__() and from them create a private dictionary _specs.
        
        The value of each key will be another dictionary with keys
        'val2arr' : bool
            If True, convert the parameter to a NumPy array using setarr.
        'units' : str
            The physical units of the parameter.
        For example, if in the derived class definition:
        def _set_specs(self):
            self._specs = { 'a' : { 'val2arr' : True, 'units' : 'mm' },
                            'b' : { 'val2arr' : False, 'units' : '' } }
        Then:
            params = {'a' : 1.0, 'b' : 2.0}
            obj = DerivedClass(params, 2)
        '''
        self._specs = { 'name' : { 'val2arr' : False, 'units' : '' },
                        'x0' : { 'val2arr' : False, 'units' : 'mm' },
                        'ds' : { 'val2arr' : True, 'units' : 'mm' },
                        'dn' : { 'val2arr' : True, 'units' : 'mm' },
                        'b' : { 'val2arr' : True, 'units' : 'mm' },
                        'c' : { 'val2arr' : True, 'units' : 'mm' },
                        'd' : { 'val2arr' : False, 'units' : 'mm' } }

    def _tile_strings(self, x):
        '''
        Replicate the variable 'x' over the strings
        
        Parameters
        ----------
        x : numpy.array
            Assuming that x is a 1D array over a set of frets, use numpy.tile
            to replicate x over the strings
            
        Returns
        -------
        retval : numpy.array
            The 2D tiled array
        '''
        return np.tile(x, (self._strings.get_count(), 1))
    
    def _tile_frets(self, x, max_fret:int):
        '''
        Replicate the variable 'x' over an array of frets
        
        Parameters
        ----------
        x : numpy.array
            Assuming that x is a 1D array over a set of strings, use numpy.tile
            to replicate x over the frets
        max_fret : int
            Tile over frets 1 to max_fret
            
        Returns
        -------
        retval : numpy.array
            The 2D tiled array
        '''
        return np.tile(x, (max_fret, 1)).T
    
    def _gamma(self, n):
        '''
        Compute the fretted frequency scale function 2**(n/12)
        
        Parameters
        ----------
        n : int, list of ints, or numpy.array of ints ...
        
        Returns
        -------
        retval : float, list of floats, or numpy.array of float64 ...
            2**(n/12)
        '''
        return 2.0**(n/12.0)

    def _l0(self, ds, dn, x0, c):
        '''
        Compute the length of the open string from saddle to nut
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Open string length in mm
        '''
        length = np.sqrt( (x0 + ds + dn)**2 + c**2 )
        return length

    def _l(self, ds, x0, b, c, n):
        '''
        Compute the length of the vibrating string from saddle to fret
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Vibrating string length in mm
        '''
        length = np.sqrt( (x0/self._gamma(n) + ds)**2 + (b + c)**2 )
        return length

    def _lp(self, ds, dn, x0, b, c, d, n):
        '''
        Compute the length of the non-vibrating string from fret to nut
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Non-vibrating string length in mm
        '''
        xn = x0 / self._gamma(n)
        ln = self._l(ds, x0, b, c, n)
        length = np.sqrt( (x0 - xn + dn - d)**2 + (b + (b + c) * d / (xn + ds))**2 )
        length += ( ln / (xn + ds)  ) * d
        return length

    def _lmc(self, ds, dn, x0, b, c, d, n):
        '''
        Compute the length of the fretted string from saddle to nut
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Fretted string length in mm
        '''
        length = self._l(ds, x0, b, c, n) + self._lp(ds, dn, x0, b, c, d, n)
        return length
    
    def _q(self, ds, dn, x0, b, c, d, n):
        '''
        Compute the differential strain of the fretted string
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Dimensionless differential strain of the fretted string
        '''
        l0 = self._l0(ds, dn, x0, c)
        return (self._lmc(ds, dn, x0, b, c, d, n) - l0) / l0
    
    def _rle(self, ds, dn, x0, b, c, n):
        '''
        Compute the resonant length error
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Resonant length error in cents
        '''
        return 1200 * np.log2( self._l0(ds, dn, x0, c) / (self._gamma(n) * self._l(ds, x0, b, c, n)) )
    
    def _mde(self, ds, dn, x0, b, c, d, n):
        '''
        Compute the mass density error
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Mass density error in cents
        '''
        return 600 * np.log2( 1 + self._q(ds, dn, x0, b, c, d, n) )
    
    def _tse(self, ds, dn, x0, b, c, d, kappa, n):
        '''
        Compute the tension error
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
         'kappa' :
            A number, a list, or an ndarray representing the (dimensionless)
            string constant (2 * R + 1)
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Tension error in cents
        '''
        return 600 * np.log2( 1 + kappa * self._q(ds, dn, x0, b, c, d, n) )

    def _bse(self, b_0, n):
        '''
        Compute the bending stiffness error
        
        Parameters
        ----------
        'b_0' :
            A number, a list, or an ndarray representing the (dimensionless)
            bending stiffness of the open string
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Bending stiffness error in cents
        '''
        return 1200 * np.log2( (1 + self._gamma(n) * b_0 + (1.0 + 0.5 * np.pi**2) * (self._gamma(n) * b_0)**2)
                             / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
        
    def _tfe(self, ds, dn, x0, b, c, d, kappa, b_0, n):
        '''
        Compute the total frequency error
        
        Parameters
        ----------
        'ds' :
            A number, a list, or an ndarray representing the saddle
            setbacks for each string in mm
        'dn' :
            A number, a list, or an ndarray representing the nut
            setbacks for each string in mm
        'x0': float
            The scale length of the guitar in mm
        'b' : 
            A number, a list, or an ndarray representing the height of
            the bottom of each string above the fret board minus the
            height of the top of the frets (in mm)
        'c' :
            A number, a list, or an ndarray representing the height of
            the top of the saddle in mm relative to the nut height b
        'd': float
            The distance in mm representing the size of the fretting finger
         'kappa' :
            A number, a list, or an ndarray representing the (dimensionless)
            string constant (2 * R + 1)
        'b_0' :
            A number, a list, or an ndarray representing the (dimensionless)
            bending stiffness of the open string
        'n': int or numpy.array of ints
            Fret number(s)
        
        Returns
        -------
        retval : A float, a list of floats, or a numpy.array of float64
            Total frequency error in cents
        '''
        rle = self._rle(ds, dn, x0, b, c, n)
        mde = self._mde(ds, dn, x0, b, c, d, n)
        tse = self._tse(ds, dn, x0, b, c, d, kappa, n)
        bse = self._bse(b_0, n)
        
        return rle + mde + tse + bse

    def _freq_shifts(self, max_fret:int=12):
        '''
        Compute the frequency shifts/errors for each string over the
        fret numbers in the range 0 to max_fret; the error for fret 0
        (the open string) is zero by definition
        '''
        x0 = self._x0
        ds = self._tile_frets(self._ds, max_fret)
        dn = self._tile_frets(self._dn, max_fret)
        b = self._tile_frets(self._b, max_fret)
        c = self._tile_frets(self._c, max_fret)
        d = self._d
        kappa = self._tile_frets(self._strings.get_props().kappa.to_numpy(), max_fret)
        b_0 = self._tile_frets(self._strings.get_props().b_0.to_numpy(), max_fret)

        fret_list = np.arange(1, max_fret + 1)
        n = self._tile_strings(fret_list)
        
        shifts = self._tfe(ds, dn, x0, b, c, d, kappa, b_0, n)
        
        open_strings = np.array(self._strings.get_count() * [0]).reshape(-1,1)                       
        
        return np.hstack((open_strings, shifts))

    def approximate(self):
        '''
        Compute the saddle and nut setbacks for each string using
        the analytic approximation
        
        Returns
        -------
        ds : numpy.array with dtype=float64
            The saddle setbacks for each string
        dn : numpy.array with dtype=float64
            The nut setbacks for each string
        '''
        x_0 = self._x0
        b = self._b
        c = self._c
        d = self._d
        kappa = self._strings.get_props().kappa.to_numpy()
        b_0 = self._strings.get_props().b_0.to_numpy()

        fret_d = kappa * 2 * (2*b + c)**2 * d / x_0**2
        ds = b_0 * x_0 + kappa * ( (b + c)**2 - 7 * b**2 ) / (4 * x_0) - fret_d
        dn = -kappa * b * (5 * b + c)/(2 * x_0) - fret_d

        self._ds = ds
        self._dn = dn
        
        return ds, dn

    def compensate(self, max_fret:int=12):
        '''
        Compute the saddle and nut setbacks for each string using
        the RMS (Nonlinear Least-Squares) Fit method
        
        Parameters
        ----------
        max_fret : int
            Include all fret numbers from 1 to max_fret in the fit;
            default = 12
        
        Returns
        -------
        ds : numpy.array with dtype=float64
            The saddle setbacks for each string
        dn : numpy.array with dtype=float64
            The nut setbacks for each string
        '''
        def sigma_k(fret_list, k):
            return np.sum((self._gamma(fret_list) - 1)**k)

        x0 = self._x0
        b = self._tile_frets(self._b, max_fret)
        c = self._tile_frets(self._c, max_fret)
        d = self._d
        kappa = self._tile_frets(self._strings.get_props().kappa.to_numpy(), max_fret)
        b_0 = self._tile_frets(self._strings.get_props().b_0.to_numpy(), max_fret)

        fret_list = np.arange(1, max_fret + 1)
        n = self._tile_strings(fret_list)
        ds = np.zeros(n.shape)
        dn = np.zeros(n.shape)
        
        mde = self._mde(ds, dn, x0, b, c, d, n)
        tse = self._tse(ds, dn, x0, b, c, d, kappa, n)
        bse = self._bse(b_0, n)
        
        # Include the residual RLE quadratic error in Z_n
        z_n = (np.log(2.0) / 1200.0) * ( mde + tse + bse 
                                       - (1200/np.log(2)) * ( self._gamma(n)**2 * (b + c)**2 - c**2 ) / (2 * x0**2) )
        
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

    def minimize(self, max_fret:int=12, approx:bool=False):
        '''
        Compute the saddle and nut setbacks for each string using
        the BFGS (Nonlinear RMS) Fit method
        
        Parameters
        ----------
        max_fret : int
            Include all fret numbers from 1 to max_fret in the fit;
            default = 12
        approx : bool
            If True, use the approximate method to compute the initial
            values of ds and dn for each string; if False (the default),
            then use compensate
        
        Returns
        -------
        ds : numpy.array with dtype=float64
            The saddle setbacks for each string
        dn : numpy.array with dtype=float64
            The nut setbacks for each string
        '''
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
        
        if approx:
            ds_0, dn_0 = self.approximate()
        else:
            ds_0, dn_0 = self.compensate(max_fret)
        x0 = self._x0
        b = self._b
        c = self._c
        d = self._d
        kappa = self._strings.get_props().kappa.to_numpy()
        b_0 = self._strings.get_props().b_0.to_numpy()

        ds = np.zeros_like(ds_0)
        dn = np.zeros_like(dn_0)
        idx_list = np.arange(0, self._strings.get_count())
        fret_list = np.arange(1, max_fret + 1)
        for idx in idx_list:
            dsdn0 = np.array([ds_0[idx], dn_0[idx]])
            args = (x0, b[idx], c[idx], d, kappa[idx], b_0[idx], fret_list)
            res = minimize(minfun, dsdn0, args)
            ds[idx] = res.x[0]
            dn[idx] = res.x[1]
            
        self._ds = ds
        self._dn = dn
    
        return ds, dn

    def plot_shifts(self, max_fret:int=12, show:bool=True, harm:list=[],
                    savepath:str=None, filename:str=None, markersize:float=9.0, alpha:float=1.0):
        '''
        Plot the frequency shifts/errors for each string 
        
        Parameters
        ----------
        max_fret : int
            Plot the shifts for each string from fret number 0 (the open string) to
            max_fret; default = 12
        show : bool
            If True (the default), show the plot of the results of the fits
        savepath : str
            A valid path to a directory / folder where the figure will be
            saved, or None (the default); if None, the figure is not saved.
        filename : str
            A valid file name (including an extension where needed) that
            will contain the saved figure, or None (the default); if None,
            the figure is not saved.
        markersize : float
            The size of the markers representing the shifts at each fret number;
            default = 9.0
        alpha : float
            The opacity of the lines connecting the points; default = 1.0
            
        Returns
        -------
        retval : float
            The RMS shift/error averaged over all frets and strings
        '''
        shifts = self._freq_shifts(max_fret)
        if bool(harm):
            zero_strings = harm[0]
            zero_frets = harm[1]
            for s, n in zip(zero_strings, zero_frets):
                shifts[s-1] -= shifts[s-1][n]
        rms = np.sqrt(np.mean(shifts**2))
        names = self._strings.get_specs().string.tolist()

        fret_list = np.arange(0, max_fret + 1)
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
        
        return rms
    
    def save_setbacks_table(self, max_fret:int=12, show:bool=True, savepath:str=None, filename:str=None):
        '''
        Show and/or save (in LaTeX format) a table of the guitar setbacks for each string
        in a particular set
        
        Parameters
        ----------
        max_fret : int
            Calculate the RMS frequency error over fret 0 (the open string)
            through max_fret; default = 12
        show : bool
            If True (the default), show the table using IPython.display.
        savepath : str
            A valid path to a directory / folder, or None; if
            None (the default), the table is not saved.
        filename : str
            A valid file name (including an appropriate LaTeX extension),
            or None; if None (the default), the table is not saved.
        '''
        dnu = self._freq_shifts(max_fret)
        rms = np.sqrt(np.mean(dnu**2, axis=1))

        df = pd.DataFrame({'String': self._strings.get_specs().string.tolist(),
                           '$\Delta S$ (mm)': self._ds.tolist(),
                           '$\Delta N$ (mm)': self._dn.tolist(),
                           '$\overline{\Delta \\nu}_\\text{rms}$ (cents)': rms.tolist()})

        formatter = {'$\Delta S$ (mm)': '{:.2f}',
                     '$\Delta N$ (mm)': '{:.2f}',
                     '$\overline{\Delta \\nu}_\\text{rms}$ (cents)': '{:.3f}'}

        tabdisp(df, formatter, show, savepath, filename)      
