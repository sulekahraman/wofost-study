import sys, os
import yaml
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import pcse
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.util import DummySoilDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.fileinput import CABOFileReader
from pcse.engine import Engine

import process_data

SOIL_PATH = os.path.abspath('us_data/soil/soils_locationswcoordinates.csv')
CALENDAR_DIR = os.path.abspath('us_data/calendar_usda/')

def run_wofost(params, wdp, agro, model='wlp'):
    wofost = process_data.get_wofost_model(model, params, wdp, agro)
    wofost.run_till_terminate()
    r = wofost.get_summary_output()[0]
    return r

def augment_wofost_out(r, latitude, longitude, year, soil_id, soil_prop, county, state, sowing_date, harvest_date):
    r['Latitude'] = latitude
    r['Longitude'] = longitude
    r['Year'] = year
    r['Soil Code'] = soil_id
    r['Soil Prop'] = soil_prop
    r['County'] = county
    r['State'] = state
    r['Sowing Date'] = sowing_date
    r['Harvest Date'] = harvest_date
    return r

def process_crop_calendar(fname):
    df = pd.read_csv(fname, usecols=['State', 'Year', 'Week Ending', 'Value'])
    df['State'] = process_data.us_state_abbrev(df['State'])
    return df

def get_crop_date(df, year, state):
    df = df[df['Year'] == year]
    df = df[df['State'] == state]
    df = df.reset_index()
    date = df.iloc[df['Value'].sub(50).abs().idxmin()]['Week Ending']
    return date

def get_crop_calendar_date(crop_name, date_desc, year, state, cal_dir=CALENDAR_DIR):
    date_file = os.path.join(cal_dir, f'{crop_name}_pct_{date_desc}.csv')
    date_df = process_crop_calendar(date_file)
    date = get_crop_date(date_df, year, state)
    return date
   
def get_crop_dates(year, state, plant_df, harvest_df):
    plant_date = get_crop_date(plant_df, year, state) # get_sowing_date(plant_df, year, state)
    harvest_date = get_crop_date(harvest_df, year, state) # get_harvest_date(harvest_df, year, state)
    return plant_date, harvest_date

def get_crop_calendar_df(crop_name, cal_dir=CALENDAR_DIR):
    plant_file = os.path.join(cal_dir, f'{crop_name}_pct_planted.csv')
    harvest_file = os.path.join(cal_dir, f'{crop_name}_pct_harvested.csv')
    
    plant_df = process_crop_calendar(plant_file)
    harvest_df = process_crop_calendar(harvest_file)
    return plant_df, harvest_df

def run_wofost_simulation(crop_name, variety_name, yield_d, start, end, out_file, model='wlp', soil_path=SOIL_PATH):
    cropd = process_data.get_crop_data(crop_name, variety_name)
    sited = process_data.get_site_data(CO2=360, WAV=100)
    print('Crop data:', cropd)
    print('Site data:', sited)

    soil_df = pd.read_csv(soil_path)
    soil_df = process_data.process_soil_data(soil_df)
    
    plant_df, harvest_df = get_crop_calendar_df(crop_name)

    for i, row in yield_d.iterrows():
        if i < start:
            continue
        if i > end:
            break
        if i % 10 == 0:
            print(f'iter: {i}/{end}')
        latitude, longitude = row['Latitude'], row['Longitude']
        year = row['Year']
        state = row['State']
        county = row['County']

        soil_id, soil_prop = row['Soil Code'], row['Soil Prop']
        soil_row = process_data.get_soil_row(soil_df, soil_id, soil_prop)
        soild = process_data.update_soil_params(soil_df, soil_row)
        
        crop_start_date, crop_end_date = get_crop_dates(year, state, plant_df, harvest_df)
        # If no crop calendar data available, skip this data point
        if crop_start_date is np.nan or crop_end_date is np.nan:
            print(f'Missing Crop Calendar. State:{state}, Year:{year}, Coords:{latitude, longitude}')
            continue
        
        params = ParameterProvider(cropdata=cropd, sitedata=sited, soildata=soild)
        wdp = NASAPowerWeatherDataProvider(latitude=latitude, longitude=longitude)
        agro = process_data.generate_agro(crop_name, variety_name, crop_start_date, crop_end_date)
        if i == start:
            print(f'Year: {year}, State: {state}, County:{county}')
            print('Agro data:', agro)
            print(f'Crop start-end date:{crop_start_date, crop_end_date}')
            print('Soil data:', soild)

        try:
            res = run_wofost(params, wdp, agro, model)
        except Exception as e:
            res = {}
            print(f'Exception at ({latitude}, {longitude}) {year}, {soil_id}, {soil_prop}: ', e)
            continue

        res = augment_wofost_out(res, latitude, longitude, year, soil_id, soil_prop,
                                 county, state, crop_start_date, crop_end_date)
        # Save to csv at each step in case the experiment is interrupted
        res_df =  pd.DataFrame(res, index=[(i, soil_id, soil_prop)])
        # If first row, write column names
        if i == start: 
            res_df.to_csv(out_file, mode='w', header=True)
        # If first row raised an exception, write the column names
        elif not os.path.exists(out_file): 
            res_df.to_csv(out_file, mode='w', header=True) 
        # Else just append the data without column names
        else:
            res_df.to_csv(out_file, mode='a', header=False)