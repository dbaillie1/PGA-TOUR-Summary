# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:32:10 2021

@author: BaillieD
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import getcwd
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import plotly.io as pio; pio.renderers.default='notebook'
import pga_tour_sum_funs# DB: Custom functions to reduce size of notebook 
from IPython.display import Markdown as md
from sklearn.linear_model import LinearRegression


def load_data(year):
    cd = getcwd()
    relPath = 'hole-' + str(year) + '.TXT'
    #dataDir = os.path.join(cd,relPath)
    
    df_hole = pd.read_csv(relPath, sep=';', encoding='latin-1')
    
    #Convert inches to feet
    #df_hole['Made Putt Distance'] = df_hole['Made Putt Distance'] * 0.0833333
    
    
    relPath = 'event-' + str(year) + '.TXT'
   # dataDir = os.path.join(cd,relPath)
    
    df_event = pd.read_csv(relPath, sep=';', encoding='latin-1')
    df_event['Player Number'] = df_event[' Player Number']


    l = df_event['Official Event(Y/N)'] == 'Y'
    #rel_events = df_event['Permanent Tournament Number'][l].unique()
    #ll = np.zeros((len(df), 1))
    #for r in rel_events:
    #    ll[df['Permanent #'] == r] = 1
    
   # df = df[ll == 1]

    return df_hole, df_event


def get_measured_data(df, min_drives):
    
    isDriving = (df['DrivingDistMeasuredFlag'] == 'Y')
    isDriving = isDriving & df['DrivingDistance_rounded_'] > 0
    dfMeas = df[isDriving]
    dfMeasMean = dfMeas.groupby('Player_').mean()
    playerCount = dfMeas.groupby('Player_').count()
    dfMeasMean['Count'] = playerCount['Tour']
    dfMeasMean = dfMeasMean[dfMeasMean['Count'] >= min_drives]
    
    return dfMeas, dfMeasMean


def get_hole_average_data(df, min_holes):
    dfHole = df
    dfHoleMean = dfHole.groupby('Player_').mean()
    playerCount = dfHole.groupby('Player_').count()
    dfHoleMean['Count'] = playerCount['Tour']
    # sum
    dfHoleSum = dfHole.groupby('Player_').sum()
                               
    
    dfHoleMean = dfHoleMean[dfHoleMean['Count'] >= min_holes]
    
    
    
    return dfHoleMean, dfHoleSum




def player_level_data(df_hole, df_event, min_events):
    #Hole
    # Get list of players
    playerIds = df_hole['Player_'].unique()
    isDriving = df_hole['DrivingDistMeasuredFlag'] == 'Y'
    isDriving = isDriving & df_hole['DrivingDistance_rounded_'] > 0
    # Create empty lists to score values
    meanPph = []
    
    for i in range(0,len(playerIds)):
    
        lp = (df_hole['Player_'] == playerIds[i])
        meanPph.append(df_hole[lp]['Putts'].mean())
        
        
    #Event
    player_ids = df_event['Player Number'].unique()
    driving_distance = []
    driving_acc = []
    money_earned = []
    name = []
    gir = []
    sg_p = []
    stroke_average = []
    pn = []
    for i in range(0,len(player_ids)):
       
        lp = (df_event['Player Number'] == player_ids[i])
        df_p = df_event[lp]
        
        if np.sum(lp) >= min_events:
            # Money
            l = list((df_p['Money'].dropna()))
            l = [x.strip(' ') for x in l]
            l = [x.replace(',','') for x in l]
            x2 = list(map(float, l))        
            money_earned.append(np.sum(x2))
            
            #Driving Distance
            driving_distance.append(df_p['Driving Distance(Total Distance)'].sum()/df_p['Driving Distance(Total Drives)'].sum())
             
            #Driving Accuracy
            driving_acc.append(100 * (df_p['Driving Acc. %(Fairways Hit)'].sum()/df_p['Driving Acc. %(Possible Fairways)'].sum()))
            
            #GIR
            gir.append(100*(df_p['Total Greens in Regulation'].sum()/df_p['Total Holes Played'].sum()))
            
            #ppr
            sg_p.append(df_p['Overall Putting Avg(# of Putts)'].sum()/df_p['Total Rounds)'].sum())
            #strokes
            stroke_average.append(df_p['Total Strokes'].sum()/df_p['Total Rounds)'].sum())
            
            
            
            #Name
            name.append(df_p['Player Name'].iloc[1])
            
            #Number
            pn.append(df_p['Player Number'].iloc[1])
            
    df_event = pd.DataFrame()
    df_event['Name'] = name
    df_event['Player Number'] = pn 
    df_event['Money'] = money_earned
    df_event['Driving Distance'] = driving_distance
    df_event['Driving Accuracy'] = driving_acc
    df_event['GIR'] = gir
    df_event['Putts Per Round'] = sg_p
    return df_event
        
    


def add_putting(df_event_player, df):
    
    
    #names = df_event_player['Name'].unique()
    names = df_event_player['Player Number'].unique()
    pph = []
    for n in names:
        #l = (df['Player Name'] == n) & (df['Event Name'] != 'World Golf Championships-Dell Technologies Match Play')
        l = (df['Player #'] == n) & (df['Event Name'] != 'World Golf Championships-Dell Technologies Match Play')
        pdat = df[l]
        pph.append(np.sum(pdat.Putts)/np.sum(l))
        
    df_event_player['Putts Per Hole'] = pph
    df_event_player['Putts Per Round'] = 18 * df_event_player['Putts Per Hole']
        
    
    
    
    return df_event_player



def getAllData(year, min_events):
    df, df_event = load_data(year)
    df_measured, df_measured_player = get_measured_data(df, 0)
    df_hole_average, df_hole_sum = get_hole_average_data(df, 0)
    df_event_player = player_level_data(df, df_event, min_events)
    
    if year == 2021:
        av = 295.7
    elif year == 2020:
        av = 296.4
    elif year == 2019:
        av = 293.3
    elif year == 2018:
        av = 296.1   
    elif year == 2017:
        av = 292.1       
    elif year == 2016:
        av = 290.0
    elif year == 2015:
        av = 289.7
    return df, df_event, df_measured, df_measured_player, df_hole_average, df_hole_sum, df_event_player, av


def bubble(df, x_var, y_var, title_str, year):
    fig = px.scatter(df, x=x_var, y=y_var, trendline = 'ols',
         color="Name", size="Money", 
                 hover_name="Name", title=title_str + str(year))

    # Do regression
    x = df[x_var].to_numpy().reshape(-1, 1)
    y = df[y_var].to_numpy().reshape(-1, 1)
    model = LinearRegression().fit(x,y)
    rsq = model.score(x, y)

    # Construct line to plot
    lx = [np.min(x), np.max(x)]
    ly = lx * model.coef_[0] + model.intercept_[0]

    # Add line and text
    fig.add_scatter(x=lx, y=ly, mode='lines')
    
    return fig, model, rsq
    
    