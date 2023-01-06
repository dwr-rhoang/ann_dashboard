from bokeh.plotting import figure, save, show
from bokeh.models import Range1d, HoverTool
from bokeh.io import export_png
from operator import index
from posixpath import dirname
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import os
import evaluateann
import datetime as dt
from panel.widgets import FloatSlider as fs
import datetime as dt
import itertools
from bokeh.palettes import Set2_5 as palette
#from trainann import output_locations

# Some hard-coded stuff for now - will move to a YAML config file
dir = os.path.dirname(os.path.realpath(__file__))
inp_template = os.path.join(dir,'ann_inp.csv')
dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['Time'])
dfinps_global = dfinps.copy()
dfouts = pd.read_csv('dsm2_hist_ec_output.csv',index_col=0, parse_dates = ['Time'])
#start_date = dt.datetime(2014, 1, 1)
#end_date = dt.datetime(2014, 12, 31)
start_date = dt.datetime(2013, 10, 1)
end_date = dt.datetime(2014, 9, 30)

scale_df1 =pd.read_csv(os.path.join(dir,'input_scale.csv'),
                       index_col=0, parse_dates = ['month'])
scale_df = scale_df1.copy()

class SliderGroup:
    def __init__(self,input_loc):
        sp = dict(start=0.80,  end=1.20, step=0.05, value=1,
                  orientation = 'vertical',direction ='rtl',
                  margin=8,height=150)
        self.input_loc = input_loc
        self.fs1 = fs(name='Jan', **sp)
        self.fs2 = fs(name='Feb', **sp)
        self.fs3 = fs(name='Mar', **sp)
        self.fs4 = fs(name='Apr', **sp)
        self.fs5 = fs(name='May', **sp)
        self.fs6 = fs(name='Jun', **sp)
        self.fs7 = fs(name='Jul', **sp)
        self.fs8 = fs(name='Aug', **sp)
        self.fs9 = fs(name='Sep', **sp)
        self.fs10 = fs(name='Oct', **sp)
        self.fs11 = fs(name='Nov', **sp)
        self.fs12 = fs(name='Dec', **sp)

        self.fs_set=[self.fs1,self.fs2,self.fs3,self.fs4,self.fs5,self.fs6,
                    self.fs7,self.fs8,self.fs9,self.fs10,self.fs11,self.fs12]

        self.kwargs = dict(fs1=self.fs1,fs2=self.fs2,fs3=self.fs3,fs4=self.fs4,
                 fs5=self.fs5,fs6=self.fs6,fs7=self.fs7,fs8=self.fs8,
                 fs9=self.fs9,fs10=self.fs10,fs11=self.fs11,fs12=self.fs12)

def scale_inputs(inp_template,input_loc,scale_df,fs1,fs2,fs3,
                 fs4,fs5,fs6,fs7,fs8,fs9,fs10,fs11,fs12):
                 
    global dfinps_global
    dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['Time'])

    scale_df.loc[1,input_loc] = fs1
    scale_df.loc[2,input_loc] = fs2
    scale_df.loc[3,input_loc] = fs3
    scale_df.loc[4,input_loc] = fs4
    scale_df.loc[5,input_loc] = fs5
    scale_df.loc[6,input_loc] = fs6
    scale_df.loc[7,input_loc] = fs7
    scale_df.loc[8,input_loc] = fs8
    scale_df.loc[9,input_loc] = fs9
    scale_df.loc[10,input_loc] = fs10
    scale_df.loc[11,input_loc] = fs11
    scale_df.loc[12,input_loc] = fs12

    for mon in scale_df.index:
        dfmod = dfinps.loc[dfinps.index.month == mon ,input_loc]*scale_df.loc[mon,input_loc]
        dfinps_global.update(dfmod, overwrite=True)
    #print(dfinps_global)
    return dfinps_global

def make_input_plot(dfinp,input_loc,start_date,end_date):
    #print(dfinp.head())
    p = figure(title = "",x_axis_type='datetime')
    p.line(source = dfinp,x='Time',y=str(input_loc), line_color = 'blue',
           line_dash = 'solid', line_width=1, legend_label=input_loc)
    p.plot_height = 400
    p.plot_width = 700
    p.x_range = Range1d(start=start_date, end=end_date)
    return p

def make_ts_plot_ANN(selected_key_stations,dfinp,start_date,end_date,
                     refresh,listener,model_kind):
    colors = itertools.cycle(palette)
    refresh = refresh
    listener = listener
    p = figure(title = selected_key_stations, x_axis_type='datetime')
    for m in model_kind:
        targ_df,pred_df = evaluateann.run_ann(selected_key_stations,dfinp,dfouts,m)
        p.line(source = targ_df,x='Time',y=str(selected_key_stations),
            line_color = 'black', line_width=1, legend_label='Historical')
        p.line(source = pred_df, x='Time', y=str(selected_key_stations),
            line_color = next(colors), line_width=1, legend_label=m)

    p.plot_height = 500
    p.plot_width = 1000
    p.legend.location = 'top_left'
    p.yaxis.axis_label = 'EC (uS/cm)'
    p.xaxis.axis_label = 'Date'
    p.legend.click_policy="hide"
    p.x_range = Range1d(start=start_date, end=end_date)

    tt = [
    ("Value:", "$y{0,0.0}"),
    ("Date:", "$x{%F}"),
    ]

    p.add_tools(HoverTool(
        tooltips = tt,
        formatters = {'$x':'datetime'}
    ))

    return p

def evaluate_ann(selected_key_stations,dfinp,start_date,end_date,
                     refresh,listener,model_kind):
    refresh = refresh
    listener = listener
    targ_df,pred_df = evaluateann.run_ann(selected_key_stations,dfinp,dfouts,model_kind[0])
    df_widget = pn.widgets.Tabulator(pred_df)
    return df_widget

def listener(e1,e2,e3,e4,e5,e6):
    e1 = e1
    e2 = e2
    e3 = e3
    e4 = e4
    e5 = e5
    e6 = e6
    return None

# Widgets

inputlocs = ['northern_flow','exports']
inputlocs_w = pn.widgets.Select(name='Input Location', options = inputlocs,
                                value = 'northern_flow')

variables = ['RSMKL008', 'RSAN032', 'RSAN037', 'RSAC092', 'SLTRM004', 'ROLD024',
             'CHVCT000', 'RSAN018', 'CHSWP003', 'CHDMC006', 'SLDUT007', 'RSAN072',
             'OLD_MID', 'RSAN058', 'ROLD059', 'RSAN007', 'RSAC081', 'SLMZU025',
             'RSAC075', 'SLMZU011', 'SLSUS012', 'SLCBN002', 'RSAC064']
             
variables_w = pn.widgets.Select(name='Output Location', options = variables, value = 'RSAC092')
model_kind_w = pn.widgets.CheckBoxGroup(
                    name='ML Model Selection', value = ['Res-LSTM'],
                    options = ['Res-LSTM','Res-GRU','LSTM', 'GRU', 'ResNet'],
                    inline=True)
#model_kind_w = ['Res-LSTM','Res-GRU']
dateselect_w = pn.widgets.DateRangeSlider(name='Date Range',
                                            start=dt.datetime(1990, 1, 1),
                                            end=dt.datetime(2019, 12, 31),
                                            value=(start_date, end_date),
                                            disabled =True)

radio_group = pn.widgets.RadioButtonGroup(name='Test Selector',
                options=['year1', 'year2', 'year3','year4','year5'], 
                button_type='success')

run_btn = pn.widgets.Button(name='Run ANN', button_type='primary')
train_btn = pn.widgets.Button(name='Train ANN', button_type='primary')
refresh_btn = pn.widgets.Button(name='Refresh Plot', button_type='default',width=50)
#adjuster_w = pn.widgets.TextInput(name = 'Test', placeholder = 'modify')

title_pane = pn.pane.Markdown('''
## DSM2 Emulator Dashboard
Disclaimer: this dashboard is a prototype to demonstrate the functionality of the
 web-based user interface, and will be hosted for a limited time during its evaluation period.
  The results generated from this tool are still under review.  
  Your feeback is appreciated!
[Leave Feeback](https://forms.gle/C6ysGxvxwqK1XY54A)
''',background='whitesmoke')
disclaimer_pane = pn.pane.Markdown('''
Test
''')
assumptions_pane = pn.pane.Markdown('''
Qi, S.; He M.; Bai Z.; Ding Z.; Sandhu, P.; Chung, F.; Namadi, P.; 
Zhou, Y.; Hoang, R.; Tom, B.; Anderson, J.; Roh, D.M. 
Novel Salinity Modeling Using Deep Learning for the Sacramento–San
Joaquin Delta of California. Water 2022, 14, 3628. 
https://doi.org/10.3390/w14223628
''')

feedback_pane = pn.pane.Markdown('''
Thank you for evaluating the DSM2 Emulator Dashboard. Your feedback and suggestions are welcome.  
[Leave Feeback](https://forms.gle/C6ysGxvxwqK1XY54A)
''',background='whitesmoke')

# Bindings

northern_flow = SliderGroup('northern_flow')
scale_northern_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = northern_flow.input_loc,inp_template = inp_template,
                           **northern_flow.kwargs)

exports = SliderGroup('exports')
scale_exp = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = exports.input_loc,inp_template = inp_template,
                           **exports.kwargs)

sjr_flow = SliderGroup('sjr_flow')
scale_sjr_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_flow.input_loc,inp_template = inp_template,
                           **sjr_flow.kwargs)

#net_delta_cu = SliderGroup('net_delta_cu')
#scale_net_delta_cu = pn.bind(scale_inputs,scale_df = scale_df,
#                           input_loc = net_delta_cu.input_loc,inp_template = inp_template,
#                           **net_delta_cu.kwargs)

sjr_vernalis_ec = SliderGroup('sjr_vernalis_ec')
scale_sjr_vernalis_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_vernalis_ec.input_loc,inp_template = inp_template,
                           **sjr_vernalis_ec.kwargs)

sac_greens_ec = SliderGroup('sac_greens_ec')
scale_sac_greens_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sac_greens_ec.input_loc,inp_template = inp_template,
                           **sac_greens_ec.kwargs)


listener_bnd = pn.bind(listener,
                       e1 = scale_northern_flow,
                       e2 = scale_exp,
                       e3 = scale_sjr_flow,
                       #e4 = scale_net_delta_cu,
                       e4 = None,
                       e5 = scale_sjr_vernalis_ec,
                       e6 = scale_sac_greens_ec)

# Dashboard Layout

dash = pn.Column(title_pane,pn.Row(

    pn.Column(pn.pane.Markdown('### ANN Inputs - Input Scaler'),
            
            pn.Tabs(
                ("Northern Flow",
                pn.Column(
                pn.Row(*northern_flow.fs_set),
                pn.bind(make_input_plot,dfinp=scale_northern_flow,input_loc='northern_flow',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("Exports",
                pn.Column(
                pn.Row(*exports.fs_set),
                pn.bind(make_input_plot,dfinp=scale_exp,input_loc='exports',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("SJR flow",
                pn.Column(
                pn.Row(*sjr_flow.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sjr_flow,input_loc='sjr_flow',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

#                ("Net Delta Consumptive Use",
#                pn.Column(
#                pn.Row(*net_delta_cu.fs_set),
#                pn.bind(make_input_plot,dfinp=scale_net_delta_cu,input_loc='net_delta_cu',
#                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("SJR Vernalis EC",
                pn.Column(
                pn.Row(*sjr_vernalis_ec.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sjr_vernalis_ec,input_loc='sjr_vernalis_ec',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("Sac Greens EC",
                pn.Column(
                pn.Row(*sac_greens_ec.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sac_greens_ec,input_loc='sac_greens_ec',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("DXC",
                pn.Column()),
            )
    ),

    pn.Column(pn.pane.Markdown('### ANN Outputs'),
    pn.Tabs(
        ('Plots',
        pn.Column(
            variables_w,
            dateselect_w,
            pn.bind(make_ts_plot_ANN,
                selected_key_stations=variables_w,
                dfinp = dfinps_global,
                start_date=dateselect_w.value[0],
                end_date=dateselect_w.value[1],
                refresh=refresh_btn, 
                listener = listener_bnd,
                model_kind = model_kind_w
            ),
            model_kind_w,
            refresh_btn
        )),

        ('Tabulated Outputs',
        pn.Column(
            pn.bind(evaluate_ann,
                selected_key_stations=variables_w,
                dfinp = dfinps_global,
                start_date=dateselect_w.value[0],
                end_date=dateselect_w.value[1],
                refresh=refresh_btn, 
                listener = listener_bnd,
                model_kind = model_kind_w
            ),
        )),
    )
    )
),
assumptions_pane,
feedback_pane,
#radio_group,
)

#dfinps_test.to_csv('dfinps_test.csv')
#dfinps_global.to_csv('dfinps_global.csv')

#dash.show(title = "DSM2 ANN Emulator Dashboard")

dash.servable(title = "DSM2 ANN Emulator Dashboard")

if __name__ == '__main__':
    dash.show(title = "DSM2 ANN Emulator Dashboard")