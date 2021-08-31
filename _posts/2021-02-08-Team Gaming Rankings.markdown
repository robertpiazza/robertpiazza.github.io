---
layout: post
title:  "Team Gaming Rankings"
date:   2021-02-08
categories: data_science
---
# Ranking My Friends on Generals

- During the course of the pandemic, I started playing a decade old game called Command&Conquer Generals:Zero Hour with my brother and a bunch of friends. 

- Out of curiousity, we started keeping track of the stats from all the games and show wins and losses for each person. 

- This is a team game though so being able to pull out an individual's overall excellence and measure of how much better the person was a little more tricky. 

- For this project, I wanted to ingest the stats, create a model for predicting each person's relative strength, and end up with a score that could be used to create more even teams for future games. 

The scores are stored in a google sheet located [here](https://docs.google.com/spreadsheets/d/1ks6mqMbTgVFkQE-rZDByKVnGH4WMRMOdLSdabeMfZaA)

The output of tableau dashboard can be found [here](https://public.tableau.com/app/profile/robert.piazza/viz/Generals/Overall) or embedded at the end of this page.


```python

#import all the libraries we'll use
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean, std

import matplotlib.pyplot as plt
from itertools import combinations

import pickle
import os.path

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

RANDOM_STATE = 42 
MAX_NUMBER_OF_GAMES = 25
```

The games were stored on a google sheet so the following functions ingest the most up-to-date data. 


```python
def gsheet_api_check(SCOPES):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL):
    creds = gsheet_api_check(SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=DATA_TO_PULL).execute()
    values = result.get('values', [])
    
    if not values:
        print('No data found.')
    else:
        rows = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                  range=DATA_TO_PULL).execute()
        data = rows.get('values')
        print("COMPLETE: Data copied")
        return data
    
def push_sheet_data(SCOPES,SPREADSHEET_ID,RANGE, DATA_TO_PUSH):
    creds = gsheet_api_check(SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    body = {
        'values': DATA_TO_PUSH
    }
    result = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID, range=RANGE,
        valueInputOption='USER_ENTERED', body=body).execute()
    data = result.get('updatedCells')
    print('{0} cells updated.'.format(data))
    
    return data    
```

## Import Data from Google Sheets


```python
    
# If modifying these scopes, delete the file token.pickle.
#SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1ks6mqMbTgVFkQE-rZDByKVnGH4WMRMOdLSdabeMfZaA'

#Pulls data from the entire spreadsheet tab.
#DATA_TO_PULL = 'Games'
#or
#Pulls data only from the specified range of cells.
DATA_TO_PULL = 'Games!A1:Q4000'
data = pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL)
games = pd.DataFrame(data[1:], columns=data[0])
games = games.set_index('Index',drop=True)
#df.head()

numeric_columns = ['Team', 'Win', 'Game', 'Units Created',
       'Units Lost', 'Units Destroyed', 'Buildings Constructed',
       'Buildings Lost', 'Buildings Destroyed', 'Supplies Collected', 'Rank',
       'Inverse Rank', 'Normalized Rank']

for col in numeric_columns:
    games[col] = pd.to_numeric(games[col]).copy()
games['Date'] = pd.to_datetime(games['Date'])
games.head()
```

    COMPLETE: Data copied
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Name</th>
      <th>Faction</th>
      <th>Team</th>
      <th>Win</th>
      <th>Game</th>
      <th>Units Created</th>
      <th>Units Lost</th>
      <th>Units Destroyed</th>
      <th>Buildings Constructed</th>
      <th>Buildings Lost</th>
      <th>Buildings Destroyed</th>
      <th>Supplies Collected</th>
      <th>Rank</th>
      <th>Inverse Rank</th>
      <th>Normalized Rank</th>
    </tr>
    <tr>
      <th>Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-02-03</td>
      <td>Matt</td>
      <td>USA</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>216.0</td>
      <td>195.0</td>
      <td>140.0</td>
      <td>47.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>360860.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-02-03</td>
      <td>Skippy</td>
      <td>USA</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>129.0</td>
      <td>132.0</td>
      <td>115.0</td>
      <td>63.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>260440.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-03</td>
      <td>Neo</td>
      <td>China</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>175.0</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>233450.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-03</td>
      <td>TVH</td>
      <td>USA</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>98.0</td>
      <td>93.0</td>
      <td>93.0</td>
      <td>47.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>211565.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-02-03</td>
      <td>Pancake</td>
      <td>China</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>122.0</td>
      <td>64.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>192521.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.625</td>
    </tr>
  </tbody>
</table>
</div>



Each row will have the player's name (which must be consistent across the dataset), what faction they played as (USA, GLA, or China), their team number (which, by convention is 1 for the winning team), whether they won (1 if so, blank if not), stats from the end game screen, then three ways of calculating rank. 

For basic familiarization, here's some of the stats from the columns:


```python
games.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Win</th>
      <th>Game</th>
      <th>Units Created</th>
      <th>Units Lost</th>
      <th>Units Destroyed</th>
      <th>Buildings Constructed</th>
      <th>Buildings Lost</th>
      <th>Buildings Destroyed</th>
      <th>Supplies Collected</th>
      <th>Rank</th>
      <th>Inverse Rank</th>
      <th>Normalized Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1962.000000</td>
      <td>961.0</td>
      <td>2017.000000</td>
      <td>1992.000000</td>
      <td>1992.000000</td>
      <td>1992.000000</td>
      <td>1992.000000</td>
      <td>1992.000000</td>
      <td>1992.000000</td>
      <td>1.992000e+03</td>
      <td>1998.000000</td>
      <td>1998.000000</td>
      <td>1998.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.544852</td>
      <td>1.0</td>
      <td>167.371344</td>
      <td>130.930723</td>
      <td>124.909137</td>
      <td>116.313755</td>
      <td>32.239960</td>
      <td>13.498996</td>
      <td>13.303715</td>
      <td>1.515431e+05</td>
      <td>3.644645</td>
      <td>4.355355</td>
      <td>0.589339</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.593417</td>
      <td>0.0</td>
      <td>107.399041</td>
      <td>109.309164</td>
      <td>136.530457</td>
      <td>132.316169</td>
      <td>22.944272</td>
      <td>16.695097</td>
      <td>17.853459</td>
      <td>1.763671e+05</td>
      <td>2.037910</td>
      <td>2.037910</td>
      <td>0.286096</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.0</td>
      <td>75.000000</td>
      <td>58.000000</td>
      <td>43.000000</td>
      <td>35.000000</td>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>5.753900e+04</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>1.0</td>
      <td>154.000000</td>
      <td>96.000000</td>
      <td>85.000000</td>
      <td>78.000000</td>
      <td>27.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>9.960250e+04</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>1.0</td>
      <td>265.000000</td>
      <td>169.000000</td>
      <td>163.000000</td>
      <td>149.000000</td>
      <td>39.000000</td>
      <td>19.000000</td>
      <td>19.000000</td>
      <td>1.808620e+05</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>1.0</td>
      <td>357.000000</td>
      <td>1038.000000</td>
      <td>2108.000000</td>
      <td>1953.000000</td>
      <td>221.000000</td>
      <td>159.000000</td>
      <td>226.000000</td>
      <td>3.142320e+06</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Basic Cleaning

There may be some rows of data that need cleaning. 
The main method of dealing with it is to eliminate the problem games since we have enough data regardless. 



```python

#Remove any row that doesn't have an index
games = games[(~games.index.isna())&(~games.Game.isna())].copy()

#Set the Game row to be integers instead of floats since we'll use it to make ranges
games.loc[:,'Game']=games.Game.astype(np.int32)

#Remove any game where team data isn't present and only include the columns up to Normalized Rank
games=games.loc[~games.Team.isna(),games.columns[0:16]].copy()

#Win's are designated with a 1 if there's a win, and are empty (NA) if it's a loss
#If it's a loss, we need to use a 0, otherwise it will throw off our average win calculations
games.loc[games.Win.isna(),'Win'] = games.loc[games.Win.isna(),'Win'].fillna(0)

#For every person calculate Win ratio and average rank from normalized rankings
for name in games.Name.unique():
    #print(name)
    games.loc[games.Name==name,'Win Ratio'] = games.loc[games.Name==name,'Win'].mean()
    games.loc[games.Name==name,'Avg Rank'] = games.loc[games.Name==name,'Normalized Rank'].mean()

#We use team 1 to designate which team won in another program to see which people win and lose the most together,
#but we need to mix this up or the computer's predictive model would take that as way to easily cheat.

for i in range( int(games.Game.max()+1)):
    #randomize the team numbers for each game
    team_1 = np.random.choice([0,1])
    team_2 = 1-team_1
    games.loc[(games.Game==i)&(games.Team==1),'Team'] = team_1
    games.loc[(games.Game==i)&(games.Team==2),'Team'] = team_2
    
games.head()    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Name</th>
      <th>Faction</th>
      <th>Team</th>
      <th>Win</th>
      <th>Game</th>
      <th>Units Created</th>
      <th>Units Lost</th>
      <th>Units Destroyed</th>
      <th>Buildings Constructed</th>
      <th>Buildings Lost</th>
      <th>Buildings Destroyed</th>
      <th>Supplies Collected</th>
      <th>Rank</th>
      <th>Inverse Rank</th>
      <th>Normalized Rank</th>
      <th>Win Ratio</th>
      <th>Avg Rank</th>
    </tr>
    <tr>
      <th>Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-02-03</td>
      <td>Matt</td>
      <td>USA</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>216.0</td>
      <td>195.0</td>
      <td>140.0</td>
      <td>47.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>360860.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.125</td>
      <td>0.606383</td>
      <td>0.470567</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-02-03</td>
      <td>Skippy</td>
      <td>USA</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>129.0</td>
      <td>132.0</td>
      <td>115.0</td>
      <td>63.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>260440.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.250</td>
      <td>0.456000</td>
      <td>0.725281</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-03</td>
      <td>Neo</td>
      <td>China</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>175.0</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>233450.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.375</td>
      <td>0.521739</td>
      <td>0.439990</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-03</td>
      <td>TVH</td>
      <td>USA</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>98.0</td>
      <td>93.0</td>
      <td>93.0</td>
      <td>47.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>211565.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.500</td>
      <td>0.439024</td>
      <td>0.708537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-02-03</td>
      <td>Pancake</td>
      <td>China</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>122.0</td>
      <td>64.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>192521.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.625</td>
      <td>0.450331</td>
      <td>0.618046</td>
    </tr>
  </tbody>
</table>
</div>



## Create New Training Data

We're now going to create synthetic training data. 
For this, we'll look at each player's statistics, figure out for this particular game, what his stats were for the last N games, take the average, and use that as their nominal stats each game, then pretend the two teams played each other and make the logistic regression model predict which team will win. 


```python
#Create a new column that contains the winning team
df =  pd.DataFrame(games.loc[games.Win==1,:].groupby('Game').mean().Team.astype(np.int32))
df.columns = ['Winning_Team']

#We'll use this for segmenting out which columns to use for predicting the winning team
prediction_columns = ['Units Created', 
                      'Units Lost', 
                      'Units Destroyed',
                      'Buildings Constructed',
                      'Buildings Lost', 
                      'Buildings Destroyed', 
                      'Supplies Collected', 
                      'Avg Rank', 
                      'Win Ratio']

games_copy = games.copy()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Winning_Team</th>
    </tr>
    <tr>
      <th>Game</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#for each player, and each game, create their average win and rank stats for the previous N games
for i in range( int(games_copy.Game.max()+1)):
    for name in games_copy.loc[games_copy.Game==i,'Name'].unique():
        name_bool=games_copy.Name==name
        game_bool=games_copy.Game==i
        games_copy.loc[(name_bool)&(game_bool),'Win Ratio'] = games_copy.loc[(name_bool)&(games_copy.Game<=i),'Win'].tail(MAX_NUMBER_OF_GAMES).mean()
        games_copy.loc[(name_bool)&(game_bool),'Avg Rank'] = games_copy.loc[(name_bool)&(games_copy.Game<=i),'Normalized Rank'].tail(MAX_NUMBER_OF_GAMES).mean()
games_copy.head()    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Name</th>
      <th>Faction</th>
      <th>Team</th>
      <th>Win</th>
      <th>Game</th>
      <th>Units Created</th>
      <th>Units Lost</th>
      <th>Units Destroyed</th>
      <th>Buildings Constructed</th>
      <th>Buildings Lost</th>
      <th>Buildings Destroyed</th>
      <th>Supplies Collected</th>
      <th>Rank</th>
      <th>Inverse Rank</th>
      <th>Normalized Rank</th>
      <th>Win Ratio</th>
      <th>Avg Rank</th>
    </tr>
    <tr>
      <th>Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-02-03</td>
      <td>Matt</td>
      <td>USA</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>216.0</td>
      <td>195.0</td>
      <td>140.0</td>
      <td>47.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>360860.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.125</td>
      <td>1.0</td>
      <td>0.125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-02-03</td>
      <td>Skippy</td>
      <td>USA</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>129.0</td>
      <td>132.0</td>
      <td>115.0</td>
      <td>63.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>260440.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.250</td>
      <td>0.0</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-03</td>
      <td>Neo</td>
      <td>China</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>175.0</td>
      <td>83.0</td>
      <td>83.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>233450.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>0.375</td>
      <td>1.0</td>
      <td>0.375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-03</td>
      <td>TVH</td>
      <td>USA</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>98.0</td>
      <td>93.0</td>
      <td>93.0</td>
      <td>47.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>211565.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.500</td>
      <td>1.0</td>
      <td>0.500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-02-03</td>
      <td>Pancake</td>
      <td>China</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>122.0</td>
      <td>64.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>192521.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.625</td>
      <td>0.0</td>
      <td>0.625</td>
    </tr>
  </tbody>
</table>
</div>



## Reduce each game to a single row of stats
- For predicting each game, we're going to sum the stats for each team, then take the difference. 
- For games with large negative numbers, this will indicate that team 0 won, for mostly positive, it would show team 1 won.
- For predicting each game, most of the stats should be added except for the game, team, and win stats. 


```python

values = games_copy.loc[games_copy.Team==1,:].iloc[:,3:].groupby('Game').agg({'Team':'mean', 
                                                           'Win':'mean', 
                                                           'Game':'mean', 
                                                           'Units Created':'sum', 
                                                           'Units Lost':'sum', 
                                                           'Units Destroyed':'sum',
                                                           'Buildings Constructed':'sum', 
                                                           'Buildings Lost':'sum', 
                                                           'Buildings Destroyed':'sum',
                                                           'Supplies Collected':'sum', 
                                                           'Rank':'sum', 
                                                           'Inverse Rank':'sum', #not used
                                                           'Normalized Rank':'sum', #not used
                                                           'Win Ratio':'sum', 
                                                           'Avg Rank':'sum' 
                                                           }) - games_copy.loc[
                                                               games_copy.Team==0,:].iloc[:,3:].groupby('Game').agg({'Team':'mean', 
                                                           'Win':'mean', 
                                                           'Game':'mean', 
                                                           'Units Created':'sum', 
                                                           'Units Lost':'sum', 
                                                           'Units Destroyed':'sum',
                                                           'Buildings Constructed':'sum', 
                                                           'Buildings Lost':'sum', 
                                                           'Buildings Destroyed':'sum',
                                                           'Supplies Collected':'sum', 
                                                           'Rank':'sum', 
                                                           'Inverse Rank':'sum', #not used
                                                           'Normalized Rank':'sum', #not used
                                                           'Win Ratio':'sum', 
                                                           'Avg Rank':'sum', 
                                                           })

#create the difference columns
diff_cols = []
for col in prediction_columns:
    column_name = col+'_diff'
    diff_cols += [column_name]
    #this really isn't needed anymore but the winning team will still be needed for training value
    df.loc[:,column_name] = values.loc[:,col]
    
#create nominal game stats based on median stats for each player    
predicted_games = []    

#make stats for each game
for game in games_copy.Game.unique():  
    #print('game ', game)
    team_values = []
    
    #make stats for each team
    for team in range(2):
        #print('team ', team)
        names = []
        #make stats for each player on this team
        games_copy.loc[(games_copy.Name==name)&(games.Game<=i),'Win'].tail(MAX_NUMBER_OF_GAMES).mean()
        for name in games_copy.loc[(games_copy.Game==game)&(games_copy.Team==team),'Name'].values:
            name_stats = games_copy.loc[(games_copy.Name==name)&(games.Game<=game)].tail(MAX_NUMBER_OF_GAMES).iloc[:,6:].median()
            name_stats['Win Ratio'] = games_copy.loc[(games_copy.Name==name)&(games_copy.Game<=game),'Win'].tail(MAX_NUMBER_OF_GAMES).mean()
            name_stats['Avg Rank'] = games_copy.loc[(games_copy.Name==name)&(games_copy.Game<=game),'Normalized Rank'].tail(MAX_NUMBER_OF_GAMES).mean()
            names += [name_stats]
            names[-1].loc['Win_avg'] = games_copy.loc[(games_copy.Name==name)&
                                                      (~games_copy.Team.isna())&
                                                      (games_copy.Game<=game)].Win.fillna(0).tail(MAX_NUMBER_OF_GAMES).mean()
            #print(name)
        #combine all the medians and sum them together
        #Summing works better than an average or median since if the teams have uneven number of players, the weight is on the side with more players

        team_values  += [pd.concat(names, axis = 1).T.sum()]
    predicted_games +=[team_values[1]-team_values[0]]
X_generated = pd.concat(predicted_games, axis = 1).T[prediction_columns]
X_generated.columns = diff_cols
X_generated.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Units Created_diff</th>
      <th>Units Lost_diff</th>
      <th>Units Destroyed_diff</th>
      <th>Buildings Constructed_diff</th>
      <th>Buildings Lost_diff</th>
      <th>Buildings Destroyed_diff</th>
      <th>Supplies Collected_diff</th>
      <th>Avg Rank_diff</th>
      <th>Win Ratio_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-90.0</td>
      <td>-79.0</td>
      <td>75.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-99050.0</td>
      <td>0.500000</td>
      <td>-4.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>199.0</td>
      <td>251.5</td>
      <td>65.0</td>
      <td>57.5</td>
      <td>-13.0</td>
      <td>45.0</td>
      <td>378716.0</td>
      <td>-0.229167</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.0</td>
      <td>50.0</td>
      <td>-96.5</td>
      <td>64.5</td>
      <td>-31.5</td>
      <td>25.5</td>
      <td>81672.5</td>
      <td>-0.638889</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>208.0</td>
      <td>202.0</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>58.5</td>
      <td>-37.0</td>
      <td>173173.5</td>
      <td>0.173611</td>
      <td>1.083333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.5</td>
      <td>162.5</td>
      <td>56.5</td>
      <td>44.5</td>
      <td>22.5</td>
      <td>12.0</td>
      <td>350781.5</td>
      <td>-1.067361</td>
      <td>1.633333</td>
    </tr>
  </tbody>
</table>
</div>



## Create the Training Data 


```python
y_cols= ['Winning_Team']
X = X_generated
y = np.ravel(df[y_cols])
y[0:5]
```




    array([0, 1, 1, 1, 1])



## Gridsearch for parameters

The following is how I decided which solver algorithm and inverse regularization strength C to use for the logistic regresssion part of the model. 


```python
solvers = ['newton-cg', 'lbfgs','liblinear', 'sag', 'saga']
Cs = [1, 3, 10, 30, 100]
parameters = {'logisticregression__solver':solvers, 'logisticregression__C':Cs}

#standard test, train split should be 20-30% held back for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#cross validation
cv = KFold(n_splits = 10, random_state=10, shuffle=True)


#create model
scaler = StandardScaler()
logreg = LogisticRegression(random_state=RANDOM_STATE)

#simple pipeline of normalizing all the stats then applying logistic regression
pipe = make_pipeline(scaler, logreg)

clf = GridSearchCV(pipe, parameters, cv=10)
clf.fit(X, y)
```


```python
#make a heatmap of the results
def make_heatmap(ax, gs):
    """Helper to make a heatmap."""
    results = pd.DataFrame.from_dict(gs.cv_results_)
    results['params_str'] = results.params.apply(str)

    scores_matrix = results.pivot(index='param_logisticregression__solver', columns='param_logisticregression__C',
                                      values='mean_test_score')

    im = ax.imshow(scores_matrix)

    ax.set_xticks(np.arange(len(Cs)))
    ax.set_xticklabels([x for x in Cs])
    ax.set_xlabel('C', fontsize=15)

    ax.set_yticks(np.arange(len(solvers)))
    ax.set_yticklabels([x for x in solvers])
    ax.set_ylabel('solver', fontsize=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('Mean Test Score', rotation=-90, va="bottom",
                       fontsize=15)


fig, axes = plt.subplots(ncols=1, sharey=True)
ax2 = axes

make_heatmap(ax2, clf)


ax2.set_title('GridSearch', fontsize=15)

plt.show()
```


    
![png](output_19_0.png)
    


## Prediction Test

Create a logistic regression model, with test and training splits, and 10 cross validation folds for determing accuracy. 

Also, since the supplies column is so much larger than the rest of the variables, we're going to normalize the data set with the StandardScaler model which brings all the columns to a normal distribution by subtracting the mean and dividing by the standard deviation. 


```python
#standard test, train split should be 20-30% held back for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#cross validation
cv = KFold(n_splits = 10, random_state=10, shuffle=True)


#create model
scaler = StandardScaler()
logreg = LogisticRegression(random_state=RANDOM_STATE,C=3, solver='lbfgs')

#simple pipeline of normalizing all the stats then applying logistic regression
pipe = make_pipeline(scaler, logreg)




pipe.fit(X_train, y_train)  # apply scaling on training data
pipe.fit(X, y)

```




    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('logisticregression',
                     LogisticRegression(C=3, random_state=42))])




```python
#score it
scores = cross_val_score(pipe, X, y, scoring = 'accuracy', cv=cv, n_jobs = -1)
print('10-fold cross validation accuracy: %.3f (%.3f stdev)' % (mean(scores), std(scores)))

plt.hist(scores)
plt.xlabel("Scores")
plt.show()
```

    10-fold cross validation accuracy: 0.761 (0.075 stdev)
    


    
![png](output_22_1.png)
    


For this run, we got a 10-fold cross validation accuracy of 78%. This is normal for the process. 

## Relative importance of each feature


```python
fig, ax = plt.subplots()
ax.bar(x = np.arange(len(pipe.steps[1][1].coef_[0])), height = -(pipe.steps[1][1].coef_[0]))
ax.set_xticks(np.arange(len(prediction_columns)))
ax.set_xticklabels(prediction_columns)
plt.xticks(rotation=-90)
plt.title('Feature Importance')
plt.show()
```


    
![png](output_25_0.png)
    


We can see that the model is rewarding the Win Ratio a lot, if the person has been winning a good deal recently, it will guess they continue to win. 

There's also some rewards for losing a lot of units but also destroying units, and less of a reward for creating a lot of units, and making a ton of supplies

## Example ranking

Let's see what the odds would be now for one team to play against some others, for instance, how likely is it that our best player, CoreDawg, is able to fend off two hard armies? 


```python
#chances of one team possibilities

first_team=['Neo', 'Shift', 'Matt', 'Skippy'] 
second_team= ['Hard', 'Hard', 'Hard', 'Hard']
test_team = [first_team, second_team]


team_values = []

for team in range(2):
    #print('team ', team)
    names = []
    for name in test_team[team]:
        #"Make sure we're getting correct names input:
        if name not in games.Name.unique():
            print(name +' not found')
            break
        name_stats = games_copy.loc[(games_copy.Name==name)].tail(MAX_NUMBER_OF_GAMES).iloc[:,6:].mean()
        name_stats['Win Ratio'] = games_copy.loc[(games_copy.Name==name)&(games_copy.Game<=game),'Win'].tail(MAX_NUMBER_OF_GAMES).mean()
        name_stats['Avg Rank'] = games_copy.loc[(games_copy.Name==name)&(games_copy.Game<=game),'Normalized Rank'].tail(MAX_NUMBER_OF_GAMES).mean()
        names += [name_stats]
        names[-1].loc['Win_avg'] = games_copy.loc[(games_copy.Name==name)&
                                                  (~games_copy.Team.isna())&
                                                  (games_copy.Game<=game)].Win.fillna(0).tail(MAX_NUMBER_OF_GAMES).mean()

    team_values  += [pd.concat(names, axis = 1).T.sum()]
predicted_games =[team_values[1]-team_values[0]]
X_predict = pd.concat(predicted_games, axis = 1).T[prediction_columns]
predicted_win = pipe.predict(X_predict)[0]
probability = pipe.predict_proba(X_predict)[0][predicted_win]

print('Between '+ ', '.join(first_team)+' and ' + ', '.join(second_team)+ ',\nModel predicts ' + ', '.join(test_team[predicted_win]) + ' with a '+"{:.2%}".format(probability)+' chance')
```

    Between Neo, Shift, Matt, Skippy and Hard, Hard, Hard, Hard,
    Model predicts Neo, Shift, Matt, Skippy with a 83.50% chance
    

## Team Generator

What we're really interested in though, is making even teams given a list of contestants. 

We'll use the `combinations` function from `itertools` to generate all possible teams that take half the players. For 8 players, there's 70 different unique combinations. However, for each of those 70, there's another unique team uses the other four players. This means we have effectively 35 different ways we can organize the players into two teams. 



```python
#all possibilities
all_names = ['Hard', 'Hard', 'Hard', 'Hard', 'Neo', 'Shift','Matt', 'pcap']

#how many combos are possible?
possible_combos = list(combinations(all_names,int(len(all_names)/2)))
non_parity_combos= int(len(possible_combos)/2)
possible_combos    
```




    [('Hard', 'Hard', 'Hard', 'Hard'),
     ('Hard', 'Hard', 'Hard', 'Neo'),
     ('Hard', 'Hard', 'Hard', 'Shift'),
     ('Hard', 'Hard', 'Hard', 'Matt'),
     ('Hard', 'Hard', 'Hard', 'pcap'),
     ('Hard', 'Hard', 'Hard', 'Neo'),
     ('Hard', 'Hard', 'Hard', 'Shift'),
     ('Hard', 'Hard', 'Hard', 'Matt'),
     ('Hard', 'Hard', 'Hard', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Hard', 'Hard', 'Neo'),
     ('Hard', 'Hard', 'Hard', 'Shift'),
     ('Hard', 'Hard', 'Hard', 'Matt'),
     ('Hard', 'Hard', 'Hard', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Neo', 'Shift', 'Matt'),
     ('Hard', 'Neo', 'Shift', 'pcap'),
     ('Hard', 'Neo', 'Matt', 'pcap'),
     ('Hard', 'Shift', 'Matt', 'pcap'),
     ('Hard', 'Hard', 'Hard', 'Neo'),
     ('Hard', 'Hard', 'Hard', 'Shift'),
     ('Hard', 'Hard', 'Hard', 'Matt'),
     ('Hard', 'Hard', 'Hard', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Neo', 'Shift', 'Matt'),
     ('Hard', 'Neo', 'Shift', 'pcap'),
     ('Hard', 'Neo', 'Matt', 'pcap'),
     ('Hard', 'Shift', 'Matt', 'pcap'),
     ('Hard', 'Hard', 'Neo', 'Shift'),
     ('Hard', 'Hard', 'Neo', 'Matt'),
     ('Hard', 'Hard', 'Neo', 'pcap'),
     ('Hard', 'Hard', 'Shift', 'Matt'),
     ('Hard', 'Hard', 'Shift', 'pcap'),
     ('Hard', 'Hard', 'Matt', 'pcap'),
     ('Hard', 'Neo', 'Shift', 'Matt'),
     ('Hard', 'Neo', 'Shift', 'pcap'),
     ('Hard', 'Neo', 'Matt', 'pcap'),
     ('Hard', 'Shift', 'Matt', 'pcap'),
     ('Hard', 'Neo', 'Shift', 'Matt'),
     ('Hard', 'Neo', 'Shift', 'pcap'),
     ('Hard', 'Neo', 'Matt', 'pcap'),
     ('Hard', 'Shift', 'Matt', 'pcap'),
     ('Neo', 'Shift', 'Matt', 'pcap')]



Now for each unique combination, we'll create the two teams, and give a score of which team is predicted to win. 


```python
possibilities = []
for combo_index in range(non_parity_combos):
    #create the two teams for this unique combination
    first_team=list(possible_combos[combo_index])
    second_team= list(possible_combos[2*non_parity_combos-combo_index-1])
    test_team = [first_team, second_team]
    
    #holders for each team's stats
    team_values = []
    
    for team in range(2):
        #print('team ', team)
        names = []
        for name in test_team[team]:
            #each person's stats from the last N games
            names += [games.loc[games.Name==name].tail(MAX_NUMBER_OF_GAMES).iloc[:,6:].mean()]
            #except for Win average which requires some extra calculations for blank spots
            names[-1].loc['Win_avg'] = games.loc[(games.Name==name)&(~games.Team.isna())].Win.fillna(0).tail(MAX_NUMBER_OF_GAMES).mean()
        #combine each team member's stats into a table for that team
        team_values  += [pd.concat(names, axis = 1).T.sum()]
    predicted_game =[team_values[1]-team_values[0]]
    X_predict = pd.concat(predicted_game, axis = 1).T[prediction_columns]
    
    team_predicted_to_win = pipe.predict(X_predict)[0]
    probability_of_winning = pipe.predict_proba(X_predict)[0][team_predicted_to_win]
    possibilities +=[{"Team 1":first_team, "Team 2":second_team, "Predicted Team": team_predicted_to_win+1, "Probability": probability_of_winning}]
all_runs = pd.DataFrame(possibilities)
all_runs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team 1</th>
      <th>Team 2</th>
      <th>Predicted Team</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Hard, Hard, Hard, Hard]</td>
      <td>[Neo, Shift, Matt, pcap]</td>
      <td>2</td>
      <td>0.965438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Hard, Hard, Hard, Neo]</td>
      <td>[Hard, Shift, Matt, pcap]</td>
      <td>2</td>
      <td>0.590536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Hard, Hard, Hard, Shift]</td>
      <td>[Hard, Neo, Matt, pcap]</td>
      <td>2</td>
      <td>0.594433</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Hard, Hard, Hard, Matt]</td>
      <td>[Hard, Neo, Shift, pcap]</td>
      <td>2</td>
      <td>0.499961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Hard, Hard, Hard, pcap]</td>
      <td>[Hard, Neo, Shift, Matt]</td>
      <td>2</td>
      <td>0.996667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[Hard, Hard, Hard, Neo]</td>
      <td>[Hard, Shift, Matt, pcap]</td>
      <td>2</td>
      <td>0.590536</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[Hard, Hard, Hard, Shift]</td>
      <td>[Hard, Neo, Matt, pcap]</td>
      <td>2</td>
      <td>0.594433</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[Hard, Hard, Hard, Matt]</td>
      <td>[Hard, Neo, Shift, pcap]</td>
      <td>2</td>
      <td>0.499961</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[Hard, Hard, Hard, pcap]</td>
      <td>[Hard, Neo, Shift, Matt]</td>
      <td>2</td>
      <td>0.996667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>1</td>
      <td>0.908797</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>1</td>
      <td>0.940660</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>2</td>
      <td>0.939488</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>1</td>
      <td>0.944947</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>2</td>
      <td>0.940093</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>2</td>
      <td>0.914317</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[Hard, Hard, Hard, Neo]</td>
      <td>[Hard, Shift, Matt, pcap]</td>
      <td>2</td>
      <td>0.590536</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[Hard, Hard, Hard, Shift]</td>
      <td>[Hard, Neo, Matt, pcap]</td>
      <td>2</td>
      <td>0.594433</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[Hard, Hard, Hard, Matt]</td>
      <td>[Hard, Neo, Shift, pcap]</td>
      <td>2</td>
      <td>0.499961</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[Hard, Hard, Hard, pcap]</td>
      <td>[Hard, Neo, Shift, Matt]</td>
      <td>2</td>
      <td>0.996667</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>1</td>
      <td>0.908797</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>1</td>
      <td>0.940660</td>
    </tr>
    <tr>
      <th>21</th>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>2</td>
      <td>0.939488</td>
    </tr>
    <tr>
      <th>22</th>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>1</td>
      <td>0.944947</td>
    </tr>
    <tr>
      <th>23</th>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>2</td>
      <td>0.940093</td>
    </tr>
    <tr>
      <th>24</th>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>2</td>
      <td>0.914317</td>
    </tr>
    <tr>
      <th>25</th>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>1</td>
      <td>0.908797</td>
    </tr>
    <tr>
      <th>26</th>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>1</td>
      <td>0.940660</td>
    </tr>
    <tr>
      <th>27</th>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>2</td>
      <td>0.939488</td>
    </tr>
    <tr>
      <th>28</th>
      <td>[Hard, Hard, Shift, Matt]</td>
      <td>[Hard, Hard, Neo, pcap]</td>
      <td>1</td>
      <td>0.944947</td>
    </tr>
    <tr>
      <th>29</th>
      <td>[Hard, Hard, Shift, pcap]</td>
      <td>[Hard, Hard, Neo, Matt]</td>
      <td>2</td>
      <td>0.940093</td>
    </tr>
    <tr>
      <th>30</th>
      <td>[Hard, Hard, Matt, pcap]</td>
      <td>[Hard, Hard, Neo, Shift]</td>
      <td>2</td>
      <td>0.914317</td>
    </tr>
    <tr>
      <th>31</th>
      <td>[Hard, Neo, Shift, Matt]</td>
      <td>[Hard, Hard, Hard, pcap]</td>
      <td>1</td>
      <td>0.986479</td>
    </tr>
    <tr>
      <th>32</th>
      <td>[Hard, Neo, Shift, pcap]</td>
      <td>[Hard, Hard, Hard, Matt]</td>
      <td>1</td>
      <td>0.545065</td>
    </tr>
    <tr>
      <th>33</th>
      <td>[Hard, Neo, Matt, pcap]</td>
      <td>[Hard, Hard, Hard, Shift]</td>
      <td>1</td>
      <td>0.639590</td>
    </tr>
    <tr>
      <th>34</th>
      <td>[Hard, Shift, Matt, pcap]</td>
      <td>[Hard, Hard, Hard, Neo]</td>
      <td>1</td>
      <td>0.639314</td>
    </tr>
  </tbody>
</table>
</div>



This gives us all the possible matchups and which teams are predicted to win and by how much

All that's left is to rank the teams, smallest chance of winning to largest to get the most even teams. In other words, if the model has difficulty guessing which team would win, they're more evenly matched. 


```python
desired_team_bool = all_runs.Probability == all_runs.Probability.min()
first_team  = all_runs.loc[desired_team_bool, 'Team 1'].values[0]
second_team = all_runs.loc[desired_team_bool, 'Team 2'].values[0]
team_predicted_to_win= all_runs.loc[desired_team_bool, 'Predicted Team'].values[0]-1

test_team = [first_team, second_team]
probability = all_runs.Probability.min()

print('\n\nFor '+ ', '.join(all_names)+',\nThe most even teams are '+ ', '.join(first_team)+' and ' + ', '.join(second_team)+ ',\nI predict ' + ', '.join(test_team[team_predicted_to_win]) + ' with a '+"{:.2%}".format(probability)+' chance')
```

    
    
    For Hard, Hard, Hard, Hard, Neo, Shift, Matt, pcap,
    The most even teams are Hard, Hard, Hard, Matt and Hard, Neo, Shift, pcap,
    I predict Hard, Neo, Shift, pcap with a 50.00% chance
    

## Update the google sheet with the updated team maker stats 

This boils down to the model's rating of each person, they can just be added together for each team and google sheets or tableau can use that number to recreate the teams generated. Because logistic regression uses the sigmoid function so the probability will shift as more ratings are added or subtracted. This boils down to trying to find the team with the most even summation of their individual ratings. 


```python
#Player rankings
names = []
stats= {}
sheets_stats = [['Name', 'Predictive Rating']]
for name in games.Name.unique():
    only_team=[name]
    predicted_games = []
    team_values = []
    for team in range(2):
        #print('team ', team)
        names = []
        for name in only_team:
            name_stats = games_copy.loc[(games_copy.Name==name)].tail(MAX_NUMBER_OF_GAMES).iloc[:,6:].median()
            name_stats['Win Ratio'] = games_copy.loc[(games_copy.Name==name),'Win'].tail(MAX_NUMBER_OF_GAMES).mean()
            name_stats['Avg Rank'] = games_copy.loc[(games_copy.Name==name),'Normalized Rank'].tail(MAX_NUMBER_OF_GAMES).mean()
            names += [name_stats]
            names[-1].loc['Win_avg'] = games_copy.loc[(games_copy.Name==name)&
                                                      (~games_copy.Team.isna())].Win.fillna(0).tail(MAX_NUMBER_OF_GAMES).mean()
        team_values  += [pd.concat(names, axis = 1).T.sum()]
    predicted_games +=[team_values[0]]
    X_predict = pd.concat(predicted_games, axis = 1).T[prediction_columns]
    #predicted_win = pipe.predict(X_predict)[0]
    probability = pipe.predict_proba(X_predict)[0][1] #the player should always be 1
    #stats[name] = -math.log((1 - probability)/probability)
    stats[name] = probability
    #sheets_stats +=[[name, -math.log((1 - probability)/probability)]]
    sheets_stats +=[[name, probability]]
    games.loc[games.Name==name, 'Predictive Rating'] = stats[name]

RANGE = "'Team Maker'!P1:Q"+str(len(sheets_stats))
push_sheet_data(SCOPES,SPREADSHEET_ID,RANGE, sheets_stats)    

#make a quick plot to ensure no one is getting a good deal
games.groupby('Name')['Predictive Rating'].max().sort_values().plot.barh()
plt.title('Ranking based on last '+str(MAX_NUMBER_OF_GAMES)+' games')
plt.show()
```

    38 cells updated.
    


    
![png](output_36_1.png)
    


## Update Tableau spreadsheet

The tableau data source will have two tabs, one of all the indivdual games which is a copy and paste of our games dataframe. 

The other tab will be a pre-calculated list of possible combinations for a 4v4, 3v3, or 2v2 game in binary form. 
This gives the possible name combinations for both the google sheet and tableau to ensure they're hitting all the name combos, create the resulting team, and rank them.  


```python
v2 = []
v3 = []
v4 = []

#Do 4v4 since it will be the longest column and we don't have to error check the other two for rows with no names and ratings
for i in range(256):
    if i == 0:
        sheets_stats[0]+=['4v4']    
    number = bin(i+1).replace('0b',"")
    if (sum([int(x) for x in number])==4) & (len(number)==8):
        v4 +=[number]
        try:
            sheets_stats[len(v4)]+=[number]
        except:
            sheets_stats+=[["", "", number]]
for i in range(64):
    if i == 0:
        sheets_stats[0]+=['3v3']    
    number = bin(i+1).replace('0b',"")
    if (sum([int(x) for x in number])==3) & (len(number)==6):
        v3 +=[number]
        sheets_stats[len(v3)]+=[number]
for i in range(16):
    if i == 0:
        sheets_stats[0]+=['2v2']
    number = bin(i+1).replace('0b',"")
    if (sum([int(x) for x in number])==2) & (len(number)==4):
        v2 +=[number]
        sheets_stats[len(v2)]+=[number]
        
pd.DataFrame(sheets_stats[1:], columns=sheets_stats[0])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Predictive Rating</th>
      <th>4v4</th>
      <th>3v3</th>
      <th>2v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Matt</td>
      <td>0.945633</td>
      <td>10000111</td>
      <td>100011</td>
      <td>1001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Skippy</td>
      <td>0.74034</td>
      <td>10001011</td>
      <td>100101</td>
      <td>1010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Neo</td>
      <td>0.872259</td>
      <td>10001101</td>
      <td>100110</td>
      <td>1100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TVH</td>
      <td>0.772892</td>
      <td>10001110</td>
      <td>101001</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pancake</td>
      <td>0.698927</td>
      <td>10010011</td>
      <td>101010</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jack</td>
      <td>0.868168</td>
      <td>10010101</td>
      <td>101100</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Scottagorn</td>
      <td>0.894642</td>
      <td>10010110</td>
      <td>110001</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SS</td>
      <td>0.339708</td>
      <td>10011001</td>
      <td>110010</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>STM</td>
      <td>0.536517</td>
      <td>10011010</td>
      <td>110100</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CoreDawg</td>
      <td>0.964444</td>
      <td>10011100</td>
      <td>111000</td>
      <td>None</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mike</td>
      <td>0.95847</td>
      <td>10100011</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Medium</td>
      <td>0.73781</td>
      <td>10100101</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Shift</td>
      <td>0.884507</td>
      <td>10100110</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Pcap</td>
      <td>0.794414</td>
      <td>10101001</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Hard</td>
      <td>0.822299</td>
      <td>10101010</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Spiff</td>
      <td>0.879811</td>
      <td>10101100</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Copper Kettle</td>
      <td>0.17295</td>
      <td>10110001</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tytan</td>
      <td>0.96384</td>
      <td>10110010</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>18</th>
      <td></td>
      <td></td>
      <td>10110100</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td></td>
      <td>10111000</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td></td>
      <td>11000011</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>21</th>
      <td></td>
      <td></td>
      <td>11000101</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>22</th>
      <td></td>
      <td></td>
      <td>11000110</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>23</th>
      <td></td>
      <td></td>
      <td>11001001</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>24</th>
      <td></td>
      <td></td>
      <td>11001010</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>25</th>
      <td></td>
      <td></td>
      <td>11001100</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>26</th>
      <td></td>
      <td></td>
      <td>11010001</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>27</th>
      <td></td>
      <td></td>
      <td>11010010</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>28</th>
      <td></td>
      <td></td>
      <td>11010100</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>29</th>
      <td></td>
      <td></td>
      <td>11011000</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td></td>
      <td>11100001</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>31</th>
      <td></td>
      <td></td>
      <td>11100010</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>32</th>
      <td></td>
      <td></td>
      <td>11100100</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>33</th>
      <td></td>
      <td></td>
      <td>11101000</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>34</th>
      <td></td>
      <td></td>
      <td>11110000</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Save the data source


```python
        
games.to_excel('Generals Statistics.xlsx',sheet_name="Games")

from openpyxl import load_workbook
workbook = load_workbook(filename="Generals Statistics.xlsx")
workbook.create_sheet('Teams')
sheet = workbook['Teams']
for i, rows in enumerate(sheets_stats):
    sheet.cell(row=i+1, column=1).value = rows[0]
    sheet.cell(row=i+1, column=2).value = rows[1]
    try:
        sheet.cell(row=i+1, column=3).value = rows[2]
    except:
        pass
    try:
        sheet.cell(row=i+1, column=4).value = rows[3]
    except:
        pass
    try:
        sheet.cell(row=i+1, column=5).value = rows[4]
    except:
        pass

#add a final name of 'None' with a value of 0 so it tableau doesn't have to have eight people every time
sheet.cell(row=len(games.Name.unique())+1, column=1).value = 'None'
sheet.cell(row=len(games.Name.unique())+1, column=2).value = 0
workbook.save(filename="Generals Statistics.xlsx")   

 
```

## Conclusion

This was a great project for taking each person's individual stats, creating some synthetic games based on them, and having a logistic regression model predict which team would win. With an ~80% accuracy rate, this could probably be improved with some more complex models but they'd be a lot less explainable as well as implementable inside of tableau and google sheets. Making this usable to the other team members when I'm not around was therefore more important than an extremely accurate model. 

If you'd like to see the tableau page of the latest stats, they can be found [here](https://public.tableau.com/app/profile/robert.piazza/viz/Generals/Overall) but I've also attempted to embed it below.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/Generals/Overall?:embed=yes&:display_count=n&:showVizHome=no" width = '1000' height = '800' scrolling='yes' ></iframe>

For a more interactive version click *[here](https://public.tableau.com/views/Generals/Overall?:embed=yes&:display_count=n&:showVizHome=no)*