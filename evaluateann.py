import os
import sys
import matplotlib.pyplot as plt
#import pickle
#import time
#import re
#import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import holoviews as hv
hv.extension('bokeh')
#import hvplot.pandas
#import panel as pn
import annutils
#from collections import defaultdict

cwd = os.getcwd()
sys.path.append(cwd)
gdrive_root_path = cwd

observed_stations_ordered_by_median = ['RSMKL008', 'RSAN032', 'RSAN037', 'RSAC092', 'SLTRM004', 'ROLD024',
                                       'CHVCT000', 'RSAN018', 'CHSWP003', 'CHDMC006', 'SLDUT007', 'RSAN072',
                                       'OLD_MID', 'RSAN058', 'ROLD059', 'RSAN007', 'RSAC081', 'SLMZU025',
                                       'RSAC075', 'SLMZU011', 'SLSUS012', 'SLCBN002', 'RSAC064']

output_stations = None
exclude_input_variable='Martinez_input'


data_file = "dsm2_ann_inputs_historical_20220506.xlsx"
num_sheets = 10
ndays=118
window_size=0
nwindows=0
data_path = os.path.join(cwd,data_file)

'''
dflist = [pd.read_excel(data_path,i,index_col=0,parse_dates=True) for i in range(num_sheets)]
df_inpout = pd.concat(dflist[0:num_sheets],axis=1).dropna(axis=0)
dfinps = df_inpout.loc[:,~df_inpout.columns.isin(dflist[num_sheets-1].columns)]
dfinps.drop(columns=exclude_input_variable,inplace=True,errors='ignore')
print(dfinps)
#dfinps_test = pd.read_csv('ann_inp.csv',index_col=0, parse_dates = ['date'])
dfouts = df_inpout.loc[:,df_inpout.columns.isin(dflist[num_sheets-1].columns)]
if output_stations is None:
    # read station names
    output_stations = list(dfouts.columns)
    name_mapping = {}
    for s in output_stations:
        for ss in observed_stations_ordered_by_median:
            if ss in s:
                name_mapping[s] = ss
    output_stations = list(name_mapping.values())
    print()
print(name_mapping)
print(output_stations)
print(dfouts)

name_mapping = {'CHDMC006-CVP INTAKE': 'CHDMC006',
                'CHSWP003-CCFB_INTAKE': 'CHSWP003',
                'CHVCT000-VICTORIA INTAKE': 'CHVCT000',
                'OLD_MID-OLD RIVER NEAR MIDDLE RIVER':'OLD_MID',
                'ROLD024-OLD RIVER AT BACON ISLAND': 'ROLD024',
                'ROLD059-OLD RIVER AT TRACY BLVD': 'ROLD059',
                'RSAC064-SACRAMENTO R AT PORT CHICAGO': 'RSAC064',
                'RSAC075-MALLARDISLAND': 'RSAC075',
                'RSAC081-COLLINSVILLE': 'RSAC081',
                'RSAC092-EMMATON': 'RSAC092',
                'RSAN007-ANTIOCH': 'RSAN007',
                'RSAN018-JERSEYPOINT': 'RSAN018',
                'RSAN032-SACRAMENTO R AT SAN ANDREAS LANDING': 'RSAN032',
                'RSAN037-SAN JOAQUIN R AT PRISONERS POINT': 'RSAN037',
                'RSAN058-ROUGH AND READY ISLAND': 'RSAN058',
                'RSAN072-SAN JOAQUIN R AT BRANDT BRIDGE': 'RSAN072',
                'RSMKL008-S FORK MOKELUMNE AT TERMINOUS': 'RSMKL008',
                'SLCBN002-CHADBOURNE SLOUGH NR SUNRISE DUCK CLUB': 'SLCBN002',
                'SLDUT007-DUTCH SLOUGH': 'SLDUT007',
                'SLMZU011-MONTEZUMA SL AT BELDONS LANDING': 'SLMZU011',
                'SLMZU025-MONTEZUMA SL AT NATIONAL STEEL': 'SLMZU025',
                'SLSUS012-SUISUN SL NEAR VOLANTI SL': 'SLSUS012',
                'SLTRM004-THREE MILE SLOUGH NR SAN JOAQUIN R': 'SLTRM004'}
'''
output_stations = ['CHDMC006', 'CHSWP003', 'CHVCT000', 'OLD_MID', 'ROLD024', 'ROLD059',
                  'RSAC064', 'RSAC075', 'RSAC081', 'RSAC092', 'RSAN007', 'RSAN018',
                  'RSAN032', 'RSAN037', 'RSAN058', 'RSAN072', 'RSMKL008', 'SLCBN002',
                  'SLDUT007', 'SLMZU011', 'SLMZU025', 'SLSUS012', 'SLTRM004']

#dfouts = pd.read_csv('dsm2_hist_ec_output.csv',index_col=0, parse_dates = ['Time'])

def mse_loss_masked(y_true, y_pred):
    squared_diff = tf.reduce_sum(tf.math.squared_difference(y_pred[y_true>0],y_true[y_true>0]))
    return squared_diff/(tf.reduce_sum(tf.cast(y_true>0, tf.float32))+0.01)

def run_ann(selected_key_station,dfinps,dfouts,model_kind):
    selected_output_variables = output_stations

    model_path_dict = {'Res-LSTM':'mtl_i118_residual_lstm_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'Res-GRU':'mtl_i118_residual_gru_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'ResNet':'mtl_i118_resnet_8_2_Tune_RSAC_RSAN_n_observed_first70',
                       'LSTM':'mtl_i118_lstm8_f_o1_Tune_RSAC_RSAN_n_observed_first70',
                       'MLP':'mtl_i18_d8_d2_o1_Tune_RSAC_RSAN_n_observed_first70',
                       'GRU':'mtl_i118_g8_f_o1_Tune_RSAC_RSAN_n_observed_first70'}

    model_path_prefix = model_path_dict[model_kind]

    print('Testing MTL ANN for %d stations: ' % len(selected_output_variables),end='')

    print([station.replace('target/','').replace('target','') 
            for station in selected_output_variables],end='\n')

    print('Load Model')
    annmodel = annutils.load_model(os.path.join(gdrive_root_path,'models', model_path_prefix),
                                    custom_objects={"mse_loss_masked": mse_loss_masked})
    print(annmodel.xscaler.min_val)
    print('Predict')
    dfp=annutils.predict(annmodel.model, dfinps, annmodel.xscaler,
                            annmodel.yscaler,columns=selected_output_variables,
                            ndays=ndays,window_size=window_size,nwindows=nwindows)

    y = dfouts.loc[:,selected_key_station].copy()
    y[y<0] = float('nan')
    targ_df = pd.DataFrame(y.iloc[(ndays+nwindows*window_size-1):])
    pred_df = pd.DataFrame(dfp.loc[:,selected_key_station])
    print()
    return targ_df,pred_df


#selected_key_stations = 'RSAC092'
#dfinps = pd.read_csv('ann_inp.csv',index_col=0, parse_dates = ['Time'])
#test_df_targ,test_df_pred = run_ann(selected_key_stations,dfinps,dfouts,'Res-LSTM')
#test_df_targ.to_csv('eval_ann_targ.csv')
#test_df_pred.to_csv('eval_ann_pred.csv')


#ax = test_df_pred.plot(label = "pred")
#test_df_targ.plot(ax=ax)
#plt.show()
#print(test_df)