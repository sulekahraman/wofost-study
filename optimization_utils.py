import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import simulate_wofost
import process_data

from pcse.util import DummySoilDataProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.models import Wofost71_PP, Wofost71_WLP_FD
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.fileinput import CABOFileReader

DATA_PATH = '../corn/simulation_output/new_calendar/iowa_pp_yield_county_wofost_Grain_maize_201.csv'
SOIL_PATH = 'actual_data/soil/soils_locationswcoordinates.csv'
YIELD_PATH = '../corn/processed_data/corn_grain_yield_iowa_county_wcoordinates.csv'

class DataLoader:
    def __init__(self, data_path=DATA_PATH, yield_path=YIELD_PATH):
        self.df = pd.read_csv(data_path)
        yield_df = pd.read_csv(yield_path)
        yield_df = process_data.convert_bu_per_acre_to_ton_per_ha(yield_df)
        self.df = self.df.merge(yield_df, how='inner', on=['Year', 'State', 'County'])
    
    def train_test_split(self, test_size=0.2):
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=42, shuffle=True)
        
    def get_data_points(self, df):
        return zip(df['State'], df['County'], df['Year'])
        
    def get_true_yield(self, df, state, county, year):
            return df[(df['State'] == state) & \
                      (df['County'] == county) & \
                      (df['Year'] == year)]['Value'].item()
    
    def get_true_yields(self, df):
        return df['Value'].to_numpy()


MODELNAME = 'pp'
OUTPUT_PATH = 'calibration_output/'
class SimulationModel:
    def __init__(self, crop_name, model_name=MODELNAME, fname=OUTPUT_PATH):
        self.model_name = model_name
        soil_df = pd.read_csv(SOIL_PATH)
        self.soil_df = process_data.process_soil_data(soil_df)
        self.crop_name = crop_name
        self.plant_df, self.harvest_df = simulate_wofost.get_crop_calendar_df(self.crop_name)
        
        cropd = process_data.get_crop_data(crop_name, variety_name='Grain_maize_201') # dummy crop data
        sited = process_data.get_site_data(CO2=360, WAV=50) # dummy site data
        soild = DummySoilDataProvider() # dummy soil data (in PP setting, doesn't matter)
        self.params = ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild) 
        self.results_df = pd.DataFrame()
        self.n_calls = 0
        self.fname = fname + f'{crop_name}_{model_name}_tsum_experiments.csv'

    def run_model(self, state, county, year, tsum1, tsum2, true_yield):
        self.params.clear_override()
        self.params.set_override('TSUM1', tsum1)
        self.params.set_override('TSUM2', tsum2)
        self.n_calls += 1

        # Weather
        latitude, longitude = process_data.get_county_coords(state, county)
        wdp = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude)
        # Soil
        if self.model_name == 'pp':
            soil_id, soil_prop = '', ''
        else:
            soil_id, soil_prop = self.get_soil_info(state, county)
            soil_row = process_data.get_soil_row(soil_df, soil_id, soil_prop)
            soild = process_data.update_soil_params(soil_df, soil_row)
        # Agro
        crop_start_date, crop_end_date = simulate_wofost.get_crop_dates(year, state, self.plant_df, self.harvest_df)
        if crop_start_date is np.nan or crop_end_date is np.nan:
            print(f'Missing Crop Calendar. State:{state}, Year:{year}, Coords:{latitude, longitude}')
            return 0
        variety_name = 'Grain_maize_201'
        agro = process_data.generate_agro(self.crop_name, variety_name, crop_start_date, crop_end_date)
        # Crop
        res_df = simulate_wofost.run_wofost(self.params, wdp, agro, self.model_name)
        res_df = simulate_wofost.augment_wofost_out(res_df, latitude, longitude, year, soil_id, soil_prop,
                                                    county, state, crop_start_date, crop_end_date)
        res_df['TWSO'] /= 1000 # Convert kg/ha to ton/ha
        res_df['True Yield'] = true_yield
        res_df['Yield Gap'] = res_df['True Yield'] - res_df['TWSO']
        res_df['TSUM1'] = tsum1
        res_df['TSUM2'] = tsum2
        self.results_df = self.results_df.append(res_df, ignore_index=True)
        return res_df['TWSO']

    def save_results(self, other_fname=None):
        if other_fname:
            self.results_df.to_csv(other_fname)
            print(f'Saved results to {other_fname}')
        else:
            self.results_df.to_csv(self.fname)
            print(f'Saved results to {self.fname}')
            
    def reset_results_df(self):
        self.results_df = pd.DataFrame()

def init_train_test_data(dl, year, interval=4):
    """For the evolutionary algorithm with sliding window experiment."""
    # Test data: (N+1)th year following the N year window, N=interval
    dl.test_df = dl.df[dl.df['Year'] == year]
    print(f'Test dataset size for {year}: {len(dl.test_df)}')
    if len(dl.test_df) == 0:
        return None
    # Training data: 4-year: year, year+1, year+2, year+3
    dl.train_df = dl.df[(dl.df['Year'] >= year-interval) & (dl.df['Year'] < year)]
    print(f'Train dataset size for {dl.train_df["Year"].unique()}: {len(dl.train_df)}')
    if len(dl.train_df) == 0:
        return None
    return dl

def init_data(year):
	dl = DataLoader()
	dl.df = dl.df[(dl.df['Year'] > year) & (dl.df['Year'] <= year+5) ]
	dl.train_test_split(test_size=0.2)
	dl.train_df = dl.train_df.reset_index()
	print(f'Dataset size: {len(dl.train_df)}')
	df = dl.train_df
	return dl, df

def init_simulation(crop_name='maize', model_name='pp'):
	fname = f'calibration_output/genetic_algo/{crop_name}_{model_name}_default.csv'
	sim = SimulationModel(crop_name, model_name, fname)
	return sim
	
def init_data_and_simulation(year, crop_name='maize', model_name='pp'):
	dl, df = init_data(year)
	sim = init_simulation(crop_name, model_name)	
	return sim, dl, df

def compute_mse(x, sim, dl, df, fname='calibration_output/default.csv', save=False):
    tsum1, tsum2 = x[0], x[1]
    pred_yield = np.zeros(len(df))
    true_yield = np.zeros(len(df))
    for i, (state, county, year) in enumerate(dl.get_data_points(df)):
        true_yield[i] = dl.get_true_yield(df, state, county, year)
        pred_yield[i] = sim.run_model(state, county, year, tsum1, tsum2, true_yield[i])
    if save:
        sim.save_results(fname)
    sim.reset_results_df()
    loss = mean_squared_error(pred_yield, true_yield)
    return round(loss, 2)

def compute_mse_from_file(fname):
    df = pd.read_csv(fname)
    return mean_squared_error(df['TWSO'], df['True Yield'])

def compute_train_test_mses(dict_best_params, model_name='pp'):
    mse_train_dict = dict()
    mse_test_dict = dict()
    for key, value in dict_best_params.items():
        tsum1, tsum2 = value[0], value[1]
        year = int(key.split('-')[0]) - 1
        print('Year:', year)
        dl, df = init_data(year)

        fname = f'calibration_output/calibrated/{model_name}/train_{tsum1}_{tsum2}_{key}.csv'
        mse = compute_mse_from_file(fname)
        print(f'Years: {key} | TSUM1: {tsum1} | TSUM2: {tsum2}| Train MSE: {mse}')
        mse_train_dict[key] = mse

        fname = f'calibration_output/calibrated/{model_name}/test_{tsum1}_{tsum2}_{key}.csv'
        mse = compute_mse_from_file(fname)
        print(f'Years: {key} | TSUM1: {tsum1} | TSUM2: {tsum2}| Test MSE: {mse}')
        mse_test_dict[key] = mse
    return mse_train_dict, mse_test_dict

def generate_new_results(dict_best_params, crop_name='maize', model_name='pp'):
    sim = init_simulation(crop_name=crop_name, model_name=model_name)
    for key, value in dict_best_params.items():
        tsum1, tsum2 = value[0], value[1]
        year = int(key.split('-')[0]) - 1
        dl, df = init_data(year)

        fname = f'calibration_output/calibrated/{model_name}/train_{tsum1}_{tsum2}_{key}.csv'
        mse = compute_mse(value, sim, dl, dl.train_df, fname=fname, save=True)
        print(f'Years: {key} | TSUM1: {tsum1} | TSUM2: {tsum2}| Train MSE: {mse}')

        fname = f'calibration_output/calibrated/{model_name}/test_{tsum1}_{tsum2}_{key}.csv'
        mse = compute_mse(value, sim, dl, dl.test_df, fname=fname, save=True)
        print(f'Years: {key} | TSUM1: {tsum1} | TSUM2: {tsum2}| Test MSE: {mse}')


