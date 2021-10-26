import yaml
from pcse.models import Wofost71_PP
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.util import WOFOST71SiteDataProvider, DummySoilDataProvider
from pcse.fileinput import CABOFileReader
from pcse.engine import Engine
from pcse.models import Wofost71_WLP_FD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import date, datetime

def example_wofost_params(verbose=False):
    # Year 2000, TSUM1, TSUM2: 1018, 778
    year = 2000
    tsum1, tsum2 = 1018, 778

    ## Define location, crop and season
    latitude, longitude = 43.2, -94.2 # some county in Iowa
    crop_name = 'maize'
    variety_name = 'Grain_maize_201'
    campaign_start_date = f'{year}-05-01'
    plant_date = f'{year}-05-15'
    harvest_date = f'{year}-11-15'
    max_duration = 300

    # Here we define the agromanagement, no irrigation or fertilization applied
    agro_yaml = """
    - {start}:
        CropCalendar:
            crop_name: {cname}
            variety_name: {vname}
            crop_start_date: {startdate}
            crop_start_type: sowing
            crop_end_date: {enddate}
            crop_end_type: maturity
            max_duration: {maxdur}
        TimedEvents: null
        StateEvents: null
    """.format(cname=crop_name, vname=variety_name, 
               start=campaign_start_date, startdate=plant_date, 
               enddate=harvest_date, maxdur=max_duration)
    agromanagement = yaml.safe_load(agro_yaml)
    if verbose:
        print(agro_yaml)

    # Parameter sets for crop, soil and site
    cropd = YAMLCropDataProvider() # Standard crop parameter library
    soild = DummySoilDataProvider() # We don't need soil for potential production, so we use dummy values
    sited = WOFOST71SiteDataProvider(WAV=50, CO2=360.)# Some site parameters

    # Retrieve all parameters in the form of a single object. 
    # In order to see all parameters for the selected crop already, we
    # synchronise data provider cropd with the crop/variety: 
    firstkey = list(agromanagement[0])[0]
    cropcalendar = agromanagement[0][firstkey]['CropCalendar'] 
    cropd.set_active_crop(cropcalendar['crop_name'], cropcalendar['variety_name'])
    if verbose:
        print(cropd)
    params = ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild)
    # Override the variety: (TSUM1, TSUM2)
    params.clear_override()
    params.set_override('TSUM1', tsum1)
    params.set_override('TSUM2', tsum2)
    if verbose:
        print(params)
    wdp = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude)
    if verbose:
        print(wdp)
    return params, wdp, agromanagement

def generate_noisy_dist(mu, sigma, n_samples, param_name,):
    '''
    Generate n_samples samples of the parameter param_name 
    from a Gaussian dist with mean mu and standard deviation sigma
    '''
    s = np.random.normal(mu, sigma, n_samples)
    sns.distplot(s, hist=False, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},)
    plt.title(f'{param_name} distribution')
    plt.xlabel(param_name)
    plt.show()
    return s


def generate_wofost_dist(param_name, input_dist, target_variable):
    '''
    Given the samples input_dist, run WOFOST with each sample,
    and output the results. Plot the distribution of the target variable.
    '''
    results = []
    for param in input_dist:
        params.clear_override()
        params.set_override(param_name, param)
        wofost = Wofost71_PP(params, wdp, agromanagement) #potential production
        wofost.run_till_terminate()
        r = wofost.get_summary_output()
        results.append(r[0][target_variable])
    df = pd.DataFrame({param_name: input_dist,
                       target_variable: results}
                     ).set_index(param_name)
    
    sns.distplot(results, hist=False, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},)
    plt.title(f'{target_variable} distribution')
    plt.xlabel(param_name)
    plt.show()
    return df, results

def get_time_series_output(output, varnames=None):
    if not varnames:
        varnames = list(output[0].keys())
    tmp = {}
    for var in varnames:
        tmp[var] = [t[var] for t in output]
    return tmp, varnames

def plot_wofost_variables(output, varnames=None, fig=None, axes=None):
    tmp, varnames = get_time_series_output(output, varnames)
    day = tmp.pop("day")
    varnames.remove("day")
    n=len(varnames)
    if not fig:
        fig, axes = plt.subplots(nrows=n//3+1, ncols=3, figsize=(15, 5*(n//3+1)))
    for var, ax in zip(varnames, axes.flatten()):
        ax.plot_date(day, tmp[var], 'b-')
        ax.set_title(var)
        fig.autofmt_xdate()
    plt.show()
    return fig, axes

def plot_dynamics(output, x, y):
    '''x and y are variable names to plot against each other'''
    tmp, varnames = get_time_series_output(output)
    plt.plot(tmp[x], tmp[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
def plot_scatter_dynamics(output, x, y, color_time=True):
    '''x and y are variable names to plot against each other'''
    tmp, varnames = get_time_series_output(output)
    if color_time:
        ts_color = np.arange(len(tmp[x]))
        plt.scatter(tmp[x], tmp[y], c=ts_color, cmap='viridis_r')
        plt.colorbar()
    else:
        plt.scatter(tmp[x], tmp[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


class WofostOutput():
    def __init__(self, wofost_output):
        self.wofost_output = wofost_output
        self.tmp, self.varnames = get_time_series_output(wofost_output)
        for key in self.tmp.keys():
            if type(self.tmp[key]) == list:
                setattr(self, key, np.array(self.tmp[key]))
            else:
                setattr(self, key, self.tmp[key]) 
