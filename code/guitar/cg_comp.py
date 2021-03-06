import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def file_path(pathname, filename):
    if (pathname is not None) and (filename is not None):
        return pathname + filename
    else:
        return None


class GuitarString(object):

    '''Collect parameters and compute properties of guitar strings
       Estimate the frequency shift (due to frequency-pulling and dispersion)
       and the round-trip time delay (due to dispersion) of each mode q.
    
    Public Methods
    --------------
    set_scale : 
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
        self._kappa = 2 * self._r + 1

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
        return self._r
    
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
    
    def set_scale(self, scale_length):
        self._scale = scale_length
        for string in self._strings:
            string.set_scale(scale_length)

    def set_r(self, r):
        for string, r_string in zip(self._strings, r):
            string.set_r(r_string)

    def get_count(self):
        return len(self._strings)
    
    def get_string_names(self):
        names = []
        for string in self._strings:
            names.append(string.get_name())
        return names

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
        for string in self._strings:
            r.append(string.get_r())
        return np.array(r)

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

    def estimate_r(self, delta_nu:list, g, q, ds, dn, x0):
        #q = ( g / (2 * x0**2) ) * ( (b + c)**2 + b**2/(g - 1) - c**2/g )
        l0 = x0 + ds + dn

        kappa = np.zeros_like(l0)
        string_num = np.arange(0, len(self._strings))
        for string, num in zip(self._strings, string_num):
            alpha = 0.5 * ( q + (g**2 - 1) * (1 + np.pi**2) * (string._radius / (2 * l0[num]))**2 )
            beta = (g - 1) * string._radius / (2 * l0[num])
            xi = -(np.log(2)/1200.0) * delta_nu[num] - ((g - 1) * ds[num] - dn[num]) / x0
            kappa[num] = ( ( -beta + np.sqrt(beta**2 - 4 * alpha * xi) ) / (2 * alpha) )**2

        #return (600.0 / np.log(2)) * (kappa + 1)
        return 0.5 * (kappa - 1)
    
    def compensate(self, g_n, q_n):
        def sigma(g_n, k):
            return np.sum((g_n - 1)**k)

        sigma_0 = sigma(g_n, 0)
        sigma_1 = sigma(g_n, 1)
        sigma_2 = sigma(g_n, 2)
        sigma_3 = sigma(g_n, 3)
        
        sigma = np.array([[sigma_2, -sigma_1], [sigma_1, -sigma_0]])
        sum_qn = 0.5 * np.sum(q_n)
        sum_gq = 0.5 * np.sum((g_n - 1) * q_n)

        idx_list = np.arange(0, self.get_count())
        kappa = self.get_kappa()
        b0 = self.get_stiffness()
        ds = np.zeros(self.get_count())
        dn = np.zeros(self.get_count())
        for string, idx in zip(self._strings, idx_list):
            b_1 = sigma_1 * b0[idx] + 0.5 * (1 + np.pi**2) * (sigma_2 + 2.0 * sigma_1) * b0[idx]**2
            b_2 = sigma_2 * b0[idx] + 0.5 * (1 + np.pi**2) * (sigma_3 + 2.0 * sigma_2) * b0[idx]**2
            rhs = np.array([[b_2 + sum_gq * kappa[idx]], [b_1 + sum_qn * kappa[idx]]])
            lhs = np.dot(np.linalg.inv(sigma), rhs)
            ds[idx] = lhs[0]
            dn[idx] = lhs[1]
        
        return ds, dn

    def save_specs_table(self, savepath, filename):
        names = self.get_string_names()
        notes = self.get_notes()
        radii = self.get_radii()
        densities = self.get_densities()
        tensions = self.get_tensions()

        df = pd.DataFrame({'String': names,
                           'Note': notes,
                           'Radius (mm)': radii.tolist(),
                           'Density ($\times 10^{-7}$ kg/mm)': (densities * 1.0e+07).tolist(),
                           'Tension (N)': tensions.tolist()})
        table_str = df.to_latex(index=False, escape=False, float_format="%.2f", column_format='ccccc')

        filepath = file_path(savepath, filename)
        if filepath is not None:
            print(table_str,  file=open(filepath, 'w'))        
    
    def save_props_table(self, dnu, savepath, filename):
        names = self.get_string_names()
        r = self.get_r()
        kappa = self.get_kappa()
        modulus = self.get_modulus()
        stiffness = self.get_stiffness()
        df = pd.DataFrame({'String': names,
                           '$\Delta \nu_{12}$ (cents)': dnu,
                           '$R$': r.tolist(),
                           '$\kappa$': kappa.tolist(),
                           '$E$ (GPa)': modulus.tolist(),
                           '$B_0$ ($\times 10^{-3}$)': (stiffness * 1.0e+03).tolist()})
        table_str = df.to_latex(index=False, escape=False, float_format="%.1f", column_format='cccccc')#,
                          #caption="Derived physical properties of the D'Addario Pro-Arte Nylon Classical Guitar Strings -- Normal Tension (EJ45). The corresponding scale length is 650 mm.",
                          #label='tbl:ej45_props')

        print(table_str)
        filepath = file_path(savepath, filename)
        if filepath is not None:
            print(table_str,  file=open(filepath, 'w'))        


class Guitar(object):
    def __init__(self, name, string_count, strings, x0, ds, dn, b, c, dbdx = 0.0, d=0.0):
        self._name = name

        assert string_count == strings.get_count(), "Guitar '{}'".format(self._name) + " requires {} strings, but {} were provided.".format(string_count, strings.get_count())
        self._string_list = np.arange(1, string_count + 1)
        self._strings = strings
    
        self._x0 = x0

        if len(ds) == 1:
            self._ds = np.array(ds * string_count)
        else:
            self._ds = np.array(ds)
        if len(dn) == 1:
            self._dn = np.array(dn * string_count)
        else:
            self._dn = np.array(dn)

        self._b = b
        self._c = c
        self._dbdx = dbdx
        self._d = d
        
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
        retstr += 'dbdx: ' + '{:.4f}\n'.format(self._dbdx)
        retstr += 'c: ' + '{:.1f} mm\n'.format(self._c)
        retstr += 'd: ' + '{:.1f} mm\n'.format(self._d)
        retstr += self._strings.__str__() + "\n"
        rmsstr = 'RMS Frequency Errors: ['#+ '{:.2f} cents\n'.format(self._rms())
        template = '{:.{prec}}, '
        rms = self._rms()
        for m in np.arange(rms.size):
            rmsstr += template.format(rms[m], prec = 2)
        retstr += rmsstr[0:-2] + ']\n'

        return retstr
    
    def _bn(self, n):
        g_n = self.gamma(n)
        return self._b - ((g_n - 1)/g_n) * self._dbdx * self._x0
    
    def _rms(self):
        fret_list = np.arange(1, 13)
        dnu = self.freq_shifts(fret_list)
        rms = np.sqrt(np.mean(dnu[:,1:]**2, axis=1))
        
        return rms

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
        bse = 1200 * np.log2( (1 + self.gamma(n_2d) * b_0 + (1.0 + 0.5 * np.pi**2) * (self.gamma(n_2d) * b_0)**2)
                              / (1 + b_0 + (1.0 + 0.5 * np.pi**2) * b_0**2) )
        
        shifts = rle + tme + bse
        
        open_strings = np.array([np.zeros(self._string_list[-1])]).T
        
        return np.hstack((open_strings, shifts))
    
    def estimate_r(self, delta_nu:list, fret:int):
        g = self.gamma(fret)
        q = self.qn(np.array([fret]))[0,0]
        r = self._strings.estimate_r(delta_nu, g, q, self._ds, self._dn, self._x0)
        return r
    
    def compensate(self, max_fret:int):
        self._ds = np.array([0.0] * self._strings.get_count())
        self._dn = np.array([0.0] * self._strings.get_count())
        
        fret_list = np.arange(1, max_fret + 1)
        g_n = self.gamma(fret_list)
        q_n = self.qn(fret_list)[0]

        ds, dn = self._strings.compensate(g_n, q_n)
        self._ds = self._x0 * ds
        self._dn = self._x0 * dn
        
        return self._ds, self._dn

    def plot_comp(self, savepath, filename):
        fret_list = np.arange(0, 13)
        shifts = self.freq_shifts(fret_list[1:])

        labelsize = 18
        fontsize = 24
        font = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : fontsize,
                }
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        plt.figure(figsize=(8.0,6.0))
        for string in self._string_list:
            plt.plot(fret_list, shifts[string-1], label='String {}'.format(string))
        plt.xlabel('FRET', fontdict=font)
        plt.ylabel('SHIFT (cents)', fontdict=font)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', labelsize=labelsize)
        plt.xlim(fret_list[0],fret_list[-1])
        plt.ylim(-5,20)
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
        template = '{}\n' + template_ds + '\n' + template_dn
        plt.text(0.35, 14.1, template.format(self._name, ds, dn),
                 fontdict=font, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.75))
        plt.grid(True)
        
        filepath = file_path(savepath, filename)
        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight')
        plt.show()

    def plot_harm(self, zero_strings, zero_frets, savepath, filename):
        fret_list = np.arange(0, 13)
        shifts = self.freq_shifts(fret_list[1:])
        for s, n in zip(zero_strings, zero_frets):
            shifts[s-1] -= shifts[s-1][n]

        rms = np.sqrt(np.mean(shifts[:,1:]**2, axis=1))
        rmsstr = 'RMS Frequency Errors (Harmonic Tuning): ['
        template = '{:.{prec}}, '
        for m in np.arange(rms.size):
            rmsstr += template.format(rms[m], prec = 2)
        print(rmsstr[0:-2] + ']\n')

        labelsize = 18
        fontsize = 24
        font = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : fontsize,
                }
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        plt.figure(figsize=(8.0,6.0))
        
        for string in self._string_list:
            plt.plot(fret_list, shifts[string-1], label='String {}'.format(string))
        plt.xlabel('FRET', fontdict=font)
        plt.ylabel('SHIFT (cents)', fontdict=font)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', labelsize=labelsize)
        plt.xlim(fret_list[0],fret_list[-1])
        plt.ylim(-10,20)
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
        template = '{}\n' + template_ds + '\n' + template_dn
        plt.text(0.35, 12.8, template.format(self._name, ds, dn),
                 fontdict=font, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.75))
        plt.grid(True)
        
        filepath = file_path(savepath, filename)
        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight')
        plt.show()
    
    def save_setbacks_table(self, savepath, filename):
        rms = self._rms()
        df = pd.DataFrame({'String': self._strings.get_string_names(),
                           '$\Delta S$ (mm)': self._ds.tolist(),
                           '$\Delta N$ (mm)': self._dn.tolist(),
                           '$\overline{\Delta \nu}_\text{rms}$ (cents)': rms.tolist()})
        table_str = df.to_latex(index=False, escape=False, float_format="%.2f", column_format='cccc')

        filepath = file_path(savepath, filename)
        if filepath is not None:
            print(table_str,  file=open(filepath, 'w'))        
    