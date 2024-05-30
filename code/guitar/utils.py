import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


def file_path(pathname, filename):
    '''
    Return a fully qualified path to a named file.
    
    Parameters
    ----------
    pathname : str
        A valid path to a directory / folder, or None.
    filename : str
        A valid file name (including an extension where needed),
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


def figdisp(fig, show, savepath, filename):
    '''
    Dispatch a figure: show and/or save
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure object
        The figure to be displayed/saved.
    show : bool
        If True, show and the close the figure; otherwise,
        close it.
    savepath : str
        A valid path to a directory / folder, or None; if
        None, the figure is not saved.
    filename : str
        A valid file name (including an extension where needed),
        or None; if None, the figure is not saved.
    '''
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
    '''
    Dispatch a table: show and/or save
    
    Parameters
    ----------
    df : pandas.DataFrame object
        The DataFrame containing the table to be displayed/saved.
    formatter : dict
        A dictionary with keys corresponding to the column names
        in df and values that are strings containing format info.
    show : bool
        If True, show the table using IPython.display.
    savepath : str
        A valid path to a directory / folder, or None; if
        None, the table is not saved.
    filename : str
        A valid file name (including an extension where needed),
        or None; if None, the table is not saved.
    '''
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
    '''
    Convert a number or list of numbers to a NumPy array of float64
    
    Parameters
    ----------
    x : number, list of numbers, or numpy array
        Convert an input to a numpy array of dtype float64
    count : int
        Length of the returned (1D) array; can be generalized to shape
        in the future

    Returns
    -------
    retval : numpy.ndarry with dtype('float64') and size = count.
    '''
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
    _set_specs :
        Dict of keys that will be present in the parameter dictionary
        required by a derived class; default = dict()
    '''

    def __init__(self, params:dict, count:int=0):
        '''Define the dictionary keys of parameters required by the
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
        '''
        Check whether there are any missing or extra parameters
            in params
        
        Parameters
        ----------
        params : dict
            Dictionary containing a subset of keys specified in
            _set_specs; if any parameters are missing from __dict__,
            they must be included.
        '''
        specs_keys = set(self._specs.keys())
        dict_keys = set(key[1:] for key in self.__dict__.keys())
        params_keys = set(params.keys())
        
        extra = sorted(params_keys - specs_keys)
        missing = sorted( (specs_keys - params_keys)
                        & (specs_keys - dict_keys) )
        
        assert not bool(extra), 'Extra keys in params dict: {}'.format(extra)
        assert not bool(missing), 'Missing keys in params dict: {}'.format(missing)

    def _arr2str(self, key, value):
        if np.all(np.abs(value - value[0]) < 1.0e-06):
            arrstr = key + ': {:.2f}'.format(value[0])
        else:
            arrstr = key + ': ' + np.array2string(value, precision=2, floatmode='fixed', separator=', ')
        
        return arrstr
            
    def __str__(self):
        ''' Return a string containing the attributes of an object derived from
            BaseClass. Example:
                class DerivedClass(BaseClass): ...
                obj = DerivedClass(...)
                print(obj)
        '''
        try:
            retstr = self._name + ' : ' + classmro(self) + '\n'
        except AttributeError:
            retstr = classmro(self) + '\n'
        
        for key, value in self._specs.items():
            if key == 'name':
                continue
            elif value['val2arr']:
                retstr += self._arr2str(key, self.__dict__['_' + key])
            else:
                retstr += key + ': {:.2f}'.format(self.__dict__['_' + key])
            retstr += ' ' + self._specs[key]['units'] + '\n'
                
        dict_keys = set(key[1:] for key in self.__dict__.keys())
        specs_keys = set(self._specs.keys())
        specs_keys.add('specs')

        for key in (dict_keys - specs_keys):
            retstr += self.__dict__['_' + key].__str__() + '\n'

        return retstr

    def __rich__(self):
        ''' Return a string containing the attributes of an object derived from
        the BaseClass using rich text. Example:
                from rich import print
                class DerivedClass(BaseClass): ...
                obj = DerivedClass(...)
                print(obj)
        '''
        try:
            retstr = ( "[bold blue]{}[/bold blue]".format(self._name)
                      + "[bold cyan] : {}[/bold cyan]\n".format(classmro(self)) )
        except AttributeError:
            retstr = "[bold cyan]{}[/bold cyan]\n".format(classmro(self))
        
        retstr += "[green]"
        for key, value in self._specs.items():
            if key == 'name':
                continue
            elif value['val2arr']:
                retstr += self._arr2str(key, self.__dict__['_' + key])
            else:
                retstr += key + ': {:.2f}'.format(self.__dict__['_' + key])
            retstr += ' ' + self._specs[key]['units'] + '\n'
        retstr += "[/green]"        
        
        dict_keys = set(key[1:] for key in self.__dict__.keys())
        specs_keys = set(self._specs.keys())
        specs_keys.add('specs')
        retstr += "[red]"
        for key in (dict_keys - specs_keys):
            retstr += self.__dict__['_' + key].__str__() + '\n'
        retstr += "[/red]"

        return retstr

    def set_params(self, params:dict, count:int=0):
        '''Walk through the list of keyword parameters, and for each one
        create a private variable with a name that is the input parameter name
        preceded by an underscore (i.e., 'varname' becomes '_varname')
        '''
        self._check_params(params)
        for key, value in params.items():
            if self._specs[key]['val2arr']:
                self.__dict__['_' + key] = setarr(value, count)
            else:
                self.__dict__['_' + key] = value
