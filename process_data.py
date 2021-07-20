import sys, os.path
import yaml
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import pcse
from pcse.models import Wofost71_PP, Wofost71_WLP_FD
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.util import DummySoilDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.fileinput import CABOFileReader
from pcse.engine import Engine

import nass # USDA NASS - used for yield data retrieval
from ast import literal_eval
from scipy import spatial
import pickle

MIN_YEAR = 1981 #  NASA POWER has no weather data available before 1981
SOIL_FILE  = os.path.abspath('default_data/soil.cab')
#----------------------------------------------------------------------------
# YIELD
#----------------------------------------------------------------------------
def query_yield_data(query, year=None):
    API_KEY = 'BD2FBC78-043C-3D2D-AAD4-803CB98493A1' # Request this key using 
    api = nass.NassApi(API_KEY)
    q = api.query()
    for k, v in query.items():
        q.filter(k, v)
    if year:
        q.filter('year', year)
    print(q.count())
    d = q.execute()
    return pd.DataFrame(d)

def process_yield_data(yield_data, min_year=MIN_YEAR):
    yield_data['State'] = us_state_abbrev(yield_data['State'])
    yield_data = yield_data[yield_data['Year'] >= min_year]    
    return yield_data

def augment_yield_data(yield_d, county_coords, out_file):
    yield_wcoords = pd.merge(yield_d, county_coords,  how='inner', on=['County','State'])
    yield_wcoords.to_csv(out_file)
    return yield_wcoords

def convert_bu_per_acre_to_ton_per_ha(yield_d, R=62.77):
    # 1 bushel/acre = 62.77 (63) kilograms/hectare 
    # source: https://www.extension.iastate.edu/agdm/wholefarm/html/c6-80.html#:~:text=Because%20a%20hectare%20is%20equal,%3D%2012%2C553%20kg%2Fha).
    yield_c = yield_d.copy()
    yield_c['Value'] = R * yield_c['Value'] / 1000
    # print('Converted yield from bu/acre to ton/ha')
    return yield_c

def convert_bu_per_acre_to_kg_per_ha(yield_d, R=62.77):
    # 1 bushel/acre = 62.77 (63) kilograms/hectare 
    # source: https://www.extension.iastate.edu/agdm/wholefarm/html/c6-80.html#:~:text=Because%20a%20hectare%20is%20equal,%3D%2012%2C553%20kg%2Fha).
    yield_c = yield_d.copy()
    yield_c['Value'] = R * yield_c['Value']
    print('Converted yield from bu/acre to kg/ha')
    return yield_c

def create_validation_data(yield_df, wofost_df, fname):
    '''Merge yield data and wofost output into one csv'''
    new_yield_df = yield_df[['County', 'State', 'Year', 'Latitude', 'Longitude', 'Soil Latitude', 'Soil Longitude', 'Soil Distance', 'Soil Code', 'Soil Prop', 'Value']]
    new_wofost_df = wofost_df[['Year', 'State', 'County', 'TWSO', 'TAGP']]
    merged_results = new_yield_df.merge(new_wofost_df,
                                        how='inner',
                                        on=['County', 'State', 'Year'],
                                        validate='1:1')
    # merged_results = new_yield_c.merge(new_wofost_d, how='inner', left_on=['County', 'State', 'Year'], right_on=['County', 'State', 'Year'], validate='1:1')
    # merged_results = merged_results.drop(columns=['Year', 'County', 'State'])
    merged_results = merged_results.rename(columns={'Value': 'Yield'})
    merged_results.to_csv(fname)

#----------------------------------------------------------------------------
# COUNTY COORDINATES AND SOIL
#----------------------------------------------------------------------------
def get_county_coords(state, county):
    county_coords_df = pd.read_csv('actual_data/others/county_coords.csv')
    df = county_coords_df[(county_coords_df['State'] == state) & (county_coords_df['County'] == county)]
    return df['Latitude'].item(), df['Longitude'].item() 

def find_closest_soils(county_coords):
    pts = list(zip(county_coords['Longitude'], county_coords['Latitude'])) # coords of counties
    print(len(pts))
    print(len(county_coords))
    print(set(pts) - (set(pts) &set(zip(county_coords['Longitude'], county_coords['Latitude']))))
    assert len(pts) == len(county_coords), print(f'pts: {len(pts)}, county: {len(county_coords)}')
    
    # Find the closest soil profile for each county
    with open('actual_data/soil/all_soil_coords.data', 'rb') as f:
        all_soil_coords = pickle.load(f)
    tree = spatial.KDTree(all_soil_coords)
    qres = tree.query(pts, k=1)
    print(f'max distance: {max(qres[0])}, min distance: {min(qres[0])}')
    
    # Get the soil ID for the closest coordinates and add soil info to the county df
    soil_df = pd.read_csv('actual_data/soil/soils_locationswcoordinates.csv')
    soil_df = process_soil_data(soil_df)
    return soil_df
	
def augment_county_data_all_prop(out_file):
    county_coords = pd.read_csv('actual_data/others/county_coords.csv')
    soil_df = find_closest_soils(county_coords) 
    s = soil_df.explode('coordinates') 
    
    county_coords['Soil Latitude'] = ''
    county_coords['Soil Longitude'] = ''
    county_coords['Soil Code'] = ''
    county_coords['Soil Prop'] = ''
    county_coords['Soil Distance'] = ''

    for i, pt in enumerate(pts):
        closest_coord = all_soil_coords[qres[1][i]]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Latitude')] = closest_coord[1]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Longitude')] = closest_coord[0]
        
        closest_soil_df = s[s['coordinates'] == closest_coord]
        closest_soil_ids = closest_soil_df['NEWSUID'].tolist()
        closest_soil_props = closest_soil_df['PROP'].tolist()
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Code')] = str(closest_soil_ids)
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Prop')] = str(closest_soil_props)

        dist_to_soil = qres[0][i]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Distance')] = dist_to_soil
    
    county_coords.to_csv(out_file)

def augment_county_data_max_prop(out_file):
    county_coords = pd.read_csv('actual_data/others/county_coords.csv')
    soil_df = find_closest_soils(county_coords) 
    s = soil_df.explode('coordinates') 
    
    county_coords['Soil Latitude'] = ''
    county_coords['Soil Longitude'] = ''
    county_coords['Soil Code'] = ''
    county_coords['Soil Prop'] = ''
    county_coords['Soil Distance'] = ''
    
    for i, pt in enumerate(pts):
        closest_coord = all_soil_coords[qres[1][i]]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Latitude')] = closest_coord[1]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Longitude')] = closest_coord[0]
        
        closest_soil_df = s[s['coordinates'] == closest_coord]
        idx = closest_soil_df['PROP'].argmax()
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Code')] = closest_soil_df.iloc[idx]['NEWSUID']
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Prop')] = closest_soil_df.iloc[idx]['PROP']

        dist_to_soil = qres[0][i]
        county_coords.iloc[i, county_coords.columns.get_loc('Soil Distance')] = dist_to_soil
    
    county_coords.to_csv(out_file)

def process_county_data(county_coords_path):
    cols = ['State', 'County', 'Latitude', 'Longitude']
    county_coords = pd.read_csv(county_coords_path, usecols=cols)
    county_coords['County'] = county_coords['County'].str.upper()
    county_coords = county_coords.dropna()
    county_coords = county_coords.drop_duplicates(subset=['County', 'State'])
    return county_coords

def get_county_coords_and_soil(county_file=None):
    if not county_file:
        county_file = 'actual_data/others/county_coords_wsoil_id_prop.csv'
    county_coords = pd.read_csv(county_file, encoding='latin-1')
    # Uncomment below two lines if using a list of props and soils.
    # county_coords['Soil Code'] = county_coords['Soil Code'].apply(lambda x: literal_eval(x))
    # county_coords['Soil Prop'] = county_coords['Soil Prop'].apply(lambda x: literal_eval(x))
    return county_coords
#----------------------------------------------------------------------------
# CROP
#----------------------------------------------------------------------------
def get_crop_data(crop_name, variety_name):
    cropd = YAMLCropDataProvider()
    cropd.set_active_crop(crop_name, variety_name)
    #cropd['CO2'] = 360. # Doesn't work, need to add the CO2 variable to the site data instead.
    return cropd

#----------------------------------------------------------------------------
# SITE
#----------------------------------------------------------------------------
def get_site_data(**kwargs):
    sited = WOFOST71SiteDataProvider(**kwargs)
    return sited

#----------------------------------------------------------------------------
# SOIL
#----------------------------------------------------------------------------
def process_soil_data(soil_df):
    soil_df = soil_df[soil_df['coordinates'] != '[]']
    soil_df['center'] = soil_df['center'].apply(lambda x: literal_eval(x))
    soil_df['coordinates'] = soil_df['coordinates'].apply(lambda x: literal_eval(x))
    return soil_df

def get_soil_row(soil_df, soil_id, prop=None):
    # Update the known params according to real data
    soil_row = soil_df[soil_df['NEWSUID'] == soil_id]
    if prop:
        soil_row = soil_row[soil_row['PROP'] == prop].iloc[0]
    return soil_row

def update_soil_params(soil_df, soil_row, default_soil_file=SOIL_FILE):
    # Default soil params
    soild = CABOFileReader(default_soil_file)
    soil_cols = ['SMW', 'SMFCF', 'K0', 'SOPE', 'KSUB']
    for col in soil_cols:
        soild[col] = soil_row[col]
    return soild

def get_soil_coords(soil_df):
    s = soil_df.explode('coordinates') # make sure coordinates col is a list, not a string!
    all_soil_coords = s['coordinates'].unique().tolist()
    with open('actual_data/soil/all_soil_coords.data', 'wb') as f:
        pickle.dump(all_soil_coords, f)
    print('# of unique soil coordinates:', len(all_soil_coords))
    return all_soil_coords

#----------------------------------------------------------------------------
# AGROMANAGEMENT
#----------------------------------------------------------------------------
def get_simple_agro_data(crop_name, variety_name, year=2006, max_duration=200):
    # Dates from agro example from PCSE documentation pg.37 
    # https://pcse.readthedocs.io/_/downloads/en/stable/pdf/ 
    campaign_start_date = f'{year}-03-01' 
    emergence_date = f'{year}-04-15'
    agro_yaml = """
    - {start}:
        CropCalendar:
            crop_name: {cname}
            variety_name: {vname}
            crop_start_date: {startdate}
            crop_start_type: sowing
            crop_end_date: 
            crop_end_type: maturity
            max_duration: {maxdur}
        TimedEvents: null
        StateEvents: null
    """.format(cname=crop_name, vname=variety_name, 
               start=campaign_start_date, startdate=emergence_date, 
               maxdur=max_duration)
    agro = yaml.safe_load(agro_yaml)
    return agro

def generate_agro(crop_name, variety_name, crop_start_date, crop_end_date='', max_duration=300):
    agro_yaml = """
    - {start}:
        CropCalendar:
            crop_name: {cname}
            variety_name: {vname}
            crop_start_date: {crop_start_date}
            crop_start_type: sowing
            crop_end_date: {crop_end_date}
            crop_end_type: harvest
            max_duration: {maxdur}
        TimedEvents: null
        StateEvents: null
    """.format(cname=crop_name,
               vname=variety_name, 
               start=crop_start_date,
               crop_start_date=crop_start_date, 
               crop_end_date=crop_end_date,
               maxdur=max_duration)
    agro = yaml.safe_load(agro_yaml)
    return agro


def generate_agro_maturity(crop_name, variety_name, crop_start_date, crop_end_date='', max_duration=300):
    agro_yaml = """
    - {start}:
        CropCalendar:
            crop_name: {cname}
            variety_name: {vname}
            crop_start_date: {crop_start_date}
            crop_start_type: sowing
            crop_end_date: 
            crop_end_type: maturity
            max_duration: {maxdur}
        TimedEvents: null
        StateEvents: null
    """.format(cname=crop_name,
               vname=variety_name, 
               start=crop_start_date,
               crop_start_date=crop_start_date,
               maxdur=max_duration)
    agro = yaml.safe_load(agro_yaml)
    return agro

#----------------------------------------------------------------------------
# WOFOST MODEL
#----------------------------------------------------------------------------
def get_wofost_model(model_desc, params, wdp, agro):
    # 1. Potential production (PP)
    if model_desc == 'pp':
        wofost = Wofost71_PP(params, wdp, agro)
    
    # 2. Water-limited production (WLP)
    elif model_desc == 'wlp':
        wofost = Wofost71_WLP_FD(params, wdp, agro)
    
    # 3. Water and nutrient-limited production (WLP-NPK)   
    elif model_desc == 'wlp-npk':
        config = os.path.join(os.getcwd(),"default_data/WLP_NPK.conf")
        wofost = Engine(params, wdp, agro, config)
    
    else:
        print('Incorrect model description. Needs to be pp, wlp, or wlp-npk.')
        return None
    
    return wofost

#----------------------------------------------------------------------------
# WOFOST OUTPUT 
#----------------------------------------------------------------------------
def get_year_data(year, yield_d, wofost_output, out_var='TWSO', color_col='County ANSI'):
    y, w, c = [], [], [] # yield, wofost_output, color
    yield_year = yield_d[yield_d['Year'] == year]
    
    for i, y_row in yield_year.iterrows():
        y_year = y_row['Year']
        y_soil_newsuid = literal_eval(y_row['Soil Code'])[0]
        y_prop = literal_eval(y_row['Soil Prop'])[0]
        mask = ((wofost_output['Year'] == y_year) & 
                (wofost_output['Soil Code'] == y_soil_newsuid) &
                (wofost_output['Soil Prop'] == y_prop))
        
        w_df = wofost_output[mask]
        if len(w_df) != 0:
            w_row = w_df.iloc[0]
            y.append(y_row['Value'])
            w.append(w_row[out_var])
            c.append(y_row[color_col])

    return y, w, c

def plot_wofost_vs_yield(z, yield_d, wofost_output, out_var='TWSO', color_col='County ANSI'):
    for year in z:
        y, w, c = get_year_data(year, yield_d, wofost_output, out_var, color_col)
        print(f'# of unique {color_col}:', len(set(c)))
        plt.figure()
        plt.scatter(y, w, c=c)
        plt.xlabel('Real Yield Value')
        plt.ylabel(f'Wofost Output ({out_var})')
        plt.title(f'Wofost {out_var} vs. Real Yield Year={year} (Coloring: {color_col})')
        plt.legend()
        print('# of data points:', len(w))



#----------------------------------------------------------------------------
# HELPER FUNCTIONS
#----------------------------------------------------------------------------
def us_state_abbrev(states):
    ''' Given states (pd.Series) with Camelcase state names return a pd.Series with state abbreviations.'''
    state_dict = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
    }
    states = states.str.title()
    def get_state(x):
        if x not in state_dict:
            return ''
        return state_dict[x]
    states = states.apply(lambda x: get_state(x))
    return states
