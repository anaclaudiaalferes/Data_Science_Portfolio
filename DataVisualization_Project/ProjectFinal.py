import plotly.offline as pyo
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.express as px
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import plotly.offline as pyo
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
import dash
import pandas as pd
import plotly.offline as pyo
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.express as px
import os

my_path = os.getcwd() + '/world-happinessNew'


app = dash.Dash()

df_2015=pd.read_csv(my_path+'/2015.csv')
df_2016=pd.read_csv(my_path+'/2016.csv')
df_2017=pd.read_csv(my_path+'/2017.csv')
df_2018=pd.read_csv(my_path+'/2018.csv')
df_2019=pd.read_csv(my_path+'/2019.csv')

df_GDPCSV = pd.read_csv(my_path+'/API_GDP/API_GDP.csv', delimiter=',',header= 2,decimal='.',quotechar='"')

df_2015 = df_2015.rename(columns={'Economy (GDP per Capita)': 'GDP per Capita', 'Health (Life Expectancy)': 'Life Expectancy',
                            'Trust (Government Corruption)':'Government Corruption'})
df_2016 = df_2016.rename(columns={'Economy (GDP per Capita)': 'GDP per Capita', 'Health (Life Expectancy)': 'Life Expectancy',
                            'Trust (Government Corruption)':'Government Corruption'})
df_2017 = df_2017.rename(columns={'Happiness.Rank':'Happiness Rank','Happiness.Score':'Happiness Score','Economy..GDP.per.Capita.':'GDP per Capita',
                                 'Health..Life.Expectancy.':'Life Expectancy','Trust..Government.Corruption.':'Government Corruption','Dystopia.Residual':'Dystopia Residual'})
df_2018 = df_2018.rename(columns={'Overall rank':'Happiness Rank','Country or region':'Country','Healthy life expectancy':'Life Expectancy',
                                 'Freedom to make life choices':'Freedom','Perceptions of corruption':'Government Corruption','Score':'Happiness Score',
                                  'GDP per capita':'GDP per Capita'})
df_2019 = df_2019.rename(columns={'Overall rank':'Happiness Rank','Country or region':'Country','Healthy life expectancy':'Life Expectancy',
                                 'Freedom to make life choices':'Freedom','Perceptions of corruption':'Government Corruption','Score':'Happiness Score',
                                  'GDP per capita':'GDP per Capita'})

figScatter =go.Figure()

##### SCATTER 1 FILIPE  NEEDS COUNTRY DROPDOWN
traceScatter = go.Scatter(x=["2015", "2016", "2017", "2018", "2019"],
                   y=[(pd.to_numeric(df_2015[df_2015['Country']=='Portugal']['Happiness Score']).values[0]),
                    (pd.to_numeric(df_2016[df_2016['Country']=='Portugal']['Happiness Score']).values[0]),
                    (pd.to_numeric(df_2017[df_2017['Country']=='Portugal']['Happiness Score']).values[0]),
                    (pd.to_numeric(df_2018[df_2018['Country']=='Portugal']['Happiness Score'])).values[0],
                    (pd.to_numeric(df_2019[df_2019['Country']=='Portugal']['Happiness Score']).values[0])],
                   name='Country Happiness Evolution')
dataScatter=[traceScatter]
layout = go.Layout(title='Country Happiness Evolution',
                   xaxis=dict(title='Year', dtick=1),
                   yaxis=dict(title='Happiness Score'),
                   template='plotly_dark')

figScatter1 = go.Figure(data=dataScatter)


##### SCATTER 2 FILIPE

traceScatter2015 = go.Scatter(x=df_2015['Happiness Score'], y = df_2015['Government Corruption'],
                   name = 'Government Trust Contribution in Happiness', mode = 'markers', text = df_2015['Country'],
                   marker = dict(color = df_2015['Life Expectancy'],
                                 size=df_2015['GDP per Capita']*15,
                                 colorscale='Jet', showscale = True, colorbar=dict(title=dict(text='Health'))))
traceScatter2016 = go.Scatter(x=df_2016['Happiness Score'], y = df_2016['Government Corruption'],
                   name = 'Government Trust Contribution in Happiness', mode = 'markers', text = df_2016['Country'],
                   marker = dict(color = df_2016['Life Expectancy'],
                                 size=df_2016['GDP per Capita']*15,
                                 colorscale='Jet', showscale = True, colorbar=dict(title=dict(text='Health'))))

traceScatter2017 = go.Scatter(x=df_2017['Happiness Score'], y = df_2017['Government Corruption'],
                   name = 'Government Trust Contribution in Happiness', mode = 'markers', text = df_2017['Country'],
                   marker = dict(color = df_2017['Life Expectancy'],
                                 size=df_2017['GDP per Capita']*15,
                                 colorscale='Jet', showscale = True, colorbar=dict(title=dict(text='Health'))))

traceScatter2018 = go.Scatter(x=df_2018['Happiness Score'], y = df_2018['Government Corruption'],
                   name = 'Government Trust Contribution in Happiness', mode = 'markers', text = df_2018['Country'],
                   marker = dict(color = df_2018['Life Expectancy'],
                                 size=df_2018['GDP per Capita']*15,
                                 colorscale='Jet', showscale = True, colorbar=dict(title=dict(text='Health'))))

traceScatter2019 = go.Scatter(x=df_2019['Happiness Score'], y = df_2019['Government Corruption'],
                   name = 'Government Trust Contribution in Happiness', mode = 'markers', text = df_2019['Country'],
                   marker = dict(color = df_2019['Life Expectancy'],
                                 size=df_2019['GDP per Capita']*15,
                                 colorscale='Jet', showscale = True, colorbar=dict(title=dict(text='Health'))))


dataScatter2 = [traceScatter2015,traceScatter2016,traceScatter2017,traceScatter2018,traceScatter2019]

figScatter2 = go.Figure(data=dataScatter2)

#####

df_2015_corr = df_2015.drop(['Happiness Rank', 'Country','Region','Standard Error','Family','Dystopia Residual'], axis=1)
df_2016_corr = df_2016.drop(['Happiness Rank', 'Country','Region','Lower Confidence Interval','Upper Confidence Interval','Family','Dystopia Residual'], axis=1)
df_2017_corr = df_2017.drop(['Happiness Rank', 'Country', 'Whisker.high', 'Whisker.low','Family','Dystopia Residual'], axis=1)
df_2018_corr = df_2018.drop(['Happiness Rank', 'Country', 'Social support'], axis=1)
df_2019_corr = df_2019.drop(['Happiness Rank', 'Country', 'Social support'], axis=1)

traceCorr2015 = go.Heatmap(
    x = df_2015_corr.columns,
    y = df_2015_corr.columns,
  z = df_2015_corr.corr(),
    type = 'heatmap',
    colorscale = 'viridis')

traceCorr2016 = go.Heatmap(
    x = df_2016_corr.columns,
    y = df_2016_corr.columns,
  z = df_2016_corr.corr(),
    type = 'heatmap',
    colorscale = 'viridis')

traceCorr2017 = go.Heatmap(
    x = df_2017_corr.columns,
    y = df_2017_corr.columns,
  z = df_2017_corr.corr(),
    type = 'heatmap',
    colorscale = 'viridis')

traceCorr2018 = go.Heatmap(
    x = df_2018_corr.columns,
    y = df_2018_corr.columns,
  z = df_2018_corr.corr(),
    type = 'heatmap',
    colorscale = 'viridis')

traceCorr2019 = go.Heatmap(
    x = df_2019_corr.columns,
    y = df_2019_corr.columns,
  z = df_2019_corr.corr(),
    type = 'heatmap',
    colorscale = 'viridis')



dataCorr = [traceCorr2015]
figCorr = go.Figure(data=dataCorr)


##### INES/FRANCISCO

####

##########

def add_region(df):
    df['Region']= ""
    for country in df['Country'].values.tolist():
        if country in df_2015['Country'].values.tolist():
            region = df_2015[df_2015['Country'].isin(["{}".format(country)])]['Region'].values[0]
            df.loc[df['Country'] == country, ['Region']] = region
        elif country in df_2016['Country'].values.tolist():
            region = df_2016[df_2016['Country'].isin(["{}".format(country)])]['Region'].values[0]
            df.loc[df['Country'] == country, ['Region']] = region

add_region(df_2017)
add_region(df_2018)
add_region(df_2019)

############ Graph 2015

df_total = pd.DataFrame(df_2015.groupby('Region')['Country'].nunique()).rename(columns={'Country':'T'})

#top and bottom countries
df1 = df_2015.iloc[:int(df_2015.shape[0]*0.25),:2]
df2 = df_2015.iloc[int(df_2015.shape[0]*0.25):int(df_2015.shape[0]*0.5),:2]
df3 = df_2015.iloc[int(df_2015.shape[0]*0.5):int(df_2015.shape[0]*0.75),:2]
df4 = df_2015.iloc[int(df_2015.shape[0]*0.75):,:2]

df1_c = pd.DataFrame(df1.groupby('Region')['Country'].nunique()).rename(columns={'Country':'1'})
df2_c = pd.DataFrame(df2.groupby('Region')['Country'].nunique()).rename(columns={'Country':'2'})
df3_c = pd.DataFrame(df3.groupby('Region')['Country'].nunique()).rename(columns={'Country':'3'})
df4_c = pd.DataFrame(df4.groupby('Region')['Country'].nunique()).rename(columns={'Country':'4'})

df_region = pd.concat([df_total, df1_c, df2_c, df3_c, df4_c], axis=1, sort=False).fillna(0)

df_cont = df_region.rename(index={'Australia and New Zealand':"Oceania",
                                   "Central and Eastern Europe":"Europe",
                                   "Eastern Asia":"Asia",
                                   "Latin America and Caribbean":"America",
                                   "Middle East and Northern Africa":"Africa",
                                   "North America":"America",
                                   "Southeastern Asia":"Asia",
                                   "Southern Asia":"Asia",
                                   "Sub-Saharan Africa":"Africa",
                                   "Western Europe": "Europe"})

df_cont = df_cont.groupby(level=0).sum()

#percentage table
df_percentage = pd.DataFrame(round(df_cont['1']/df_cont['T'],2)*100, columns = ['df1_p'])
df_percentage['df2_p']= round(df_cont['2']/df_cont['T'],2)*100
df_percentage['df3_p']=round(df_cont['3']/df_cont['T'],2)*100
df_percentage['df4_p']=round(df_cont['4']/df_cont['T'],2)*100

#traces
t1 = {
    "name": "Africa",
    "type": "bar",
    "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
    "y": df_percentage.iloc[0,:].values.tolist(),
    "xaxis": "x",
    "yaxis": "y"
}

t2 = {
    "name": "America",
    "type": "bar",
    "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
    "y": df_percentage.iloc[1,:].values.tolist(),
    "xaxis": "x",
    "yaxis": "y"
}

t3 = {
    "name": "Asia",
    "type": "bar",
    "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
    "y": df_percentage.iloc[2,:].values.tolist(),
    "xaxis": "x",
    "yaxis": "y"
}

t4 = {
    "name": "Europe",
    "type": "bar",
    "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
    "y": df_percentage.iloc[3,:].values.tolist(),
    "xaxis": "x",
    "yaxis": "y"
}

t5 = {
    "name": "Oceania",
    "type": "bar",
    "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
    "y": df_percentage.iloc[4,:].values.tolist(),
    "xaxis": "x",
    "yaxis": "y"
}

dataRanking = [t1,t2,t3,t4,t5]

layout = go.Layout(
    title = 'Percentages by continent by ranking groups',
    xaxis = dict(title='Happiness ranking groups'),
    yaxis = dict(title='Percentage'),
    barmode = "group"
)

figRankingGroups = go.Figure(data=dataRanking, layout=layout)

#####


#countries
last = df_2015.iloc[-1:, df_2015.columns.get_loc("Country")].values[0]
penult = df_2015.iloc[-2:-1, df_2015.columns.get_loc("Country")].values[0]
second = df_2015.iloc[1:2, df_2015.columns.get_loc("Country")].values[0]
first = df_2015.iloc[:1, df_2015.columns.get_loc("Country")].values[0]

trace1comp = {
    "name": "Happiness Score",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['Happiness Score']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['Happiness Score']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['Happiness Score']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['Happiness Score']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['Happiness Score']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}

trace2comp = {
    "name": "GDP per Capita",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['GDP per Capita']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['GDP per Capita']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['GDP per Capita']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['GDP per Capita']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['GDP per Capita']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}

trace3comp = {
    "name": "Life Expectancy",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['Life Expectancy']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['Life Expectancy']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['Life Expectancy']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['Life Expectancy']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['Life Expectancy']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}


trace4comp = {
    "name": "Freedom",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['Freedom']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['Freedom']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['Freedom']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['Freedom']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['Freedom']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}

trace5comp = {
    "name": "Government Corruption",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['Government Corruption']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['Government Corruption']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['Government Corruption']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['Government Corruption']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['Government Corruption']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}


trace6comp = {
    "name": "Generosity",
    "type": "bar",
    "x": [last, penult, "Portugal", second, first],
    "y": [df_2015.loc[df_2015['Country'] == last, ['Generosity']].values[0][0],
          df_2015.loc[df_2015['Country'] == penult, ['Generosity']].values[0][0],
          df_2015.loc[df_2015['Country'] == 'Portugal', ['Generosity']].values[0][0],
          df_2015.loc[df_2015['Country'] == second, ['Generosity']].values[0][0],
          df_2015.loc[df_2015['Country'] == first, ['Generosity']].values[0][0]],
    "xaxis": "x",
    "yaxis": "y"
}


data = [trace1comp, trace2comp, trace3comp, trace4comp, trace5comp, trace6comp]


layout = go.Layout(
    title = 'Group Bar plot',
    xaxis = dict(title='Country'),
    yaxis = dict(title='Score'),
    barmode = "group"
)


figBar = go.Figure(data=data, layout=layout)

#####



##### CLAUDIA 1
dfInfo2015 = pd.DataFrame(data =df_2015)
for col in dfInfo2015.columns:
    dfInfo2015[col] = dfInfo2015[col].astype(str)

dfInfo2015['text'] = dfInfo2015['Country'] + '<br>' + \
    '.Rank : ' + dfInfo2015['Happiness Rank'] + '<br>' + \
    '.Economy : ' + dfInfo2015['GDP per Capita'] + '<br>' + \
    '.Health : ' + dfInfo2015['Life Expectancy'] + '<br>' + \
    '.Freedom : ' + dfInfo2015['Freedom'] + '<br>' + \
    '.Government Corruption : ' + dfInfo2015['Government Corruption'] + '<br>' + \
    '.Generosity : ' + dfInfo2015['Generosity']

trace_2015=go.Choropleth(locations=dfInfo2015['Country'], # Spatial coordinates
                         z = dfInfo2015['Happiness Score'].astype(float), # Data to be color-coded
                         locationmode = 'country names', # set of locations match entries in `locations`
                         colorscale = 'RdBu',
                         autocolorscale=False,
                         text=dfInfo2015['text'],
                         colorbar_title = "Happiness")

dfInfo2016 = pd.DataFrame(data =df_2016)
for col in dfInfo2016.columns:
    dfInfo2016[col] = dfInfo2016[col].astype(str)

dfInfo2016['text'] = dfInfo2016['Country'] + '<br>' + \
    '.Rank : ' + dfInfo2016['Happiness Rank']  + '<br>' + \
    '.Economy : ' + dfInfo2016['GDP per Capita'] + '<br>' + \
    '.Health : ' + dfInfo2016['Life Expectancy'] + '<br>' + \
    '.Freedom : ' + dfInfo2016['Freedom']  + '<br>' + \
    '.Government Corruption : ' + dfInfo2016['Government Corruption'] + '<br>' + \
    '.Generosity : ' + dfInfo2016['Generosity']

trace_2016=go.Choropleth(locations=dfInfo2016['Country'], # Spatial coordinates
                         z = dfInfo2016['Happiness Score'].astype(float), # Data to be color-coded
                         locationmode = 'country names', # set of locations match entries in `locations`
                         colorscale = 'RdBu',
                         autocolorscale=False,
                         text=dfInfo2016['text'],
                         colorbar_title = "Happiness")

dfInfo2017 = pd.DataFrame(data =df_2017)
for col in dfInfo2017.columns:
    dfInfo2017[col] = dfInfo2017[col].astype(str)

dfInfo2017['text'] = dfInfo2017['Country'] + '<br>' + \
    '.Rank : ' + dfInfo2017['Happiness Rank']  + '<br>' + \
    '.Economy : ' + dfInfo2017['GDP per Capita'] + '<br>' + \
    '.Health : ' + dfInfo2017['Life Expectancy'] + '<br>' + \
    '.Freedom : ' + dfInfo2017['Freedom']  + '<br>' + \
    '.Government Corruption : ' + dfInfo2017['Government Corruption'] + '<br>' + \
    '.Generosity : ' + dfInfo2017['Generosity']

trace_2017=go.Choropleth(locations=dfInfo2017['Country'], # Spatial coordinates
                         z = dfInfo2017['Happiness Score'].astype(float), # Data to be color-coded
                         locationmode = 'country names', # set of locations match entries in `locations`
                         colorscale = 'RdBu',
                         autocolorscale=False,
                         text=dfInfo2017['text'],
                         colorbar_title = "Happiness")

dfInfo2018 = pd.DataFrame(data =df_2018)
for col in dfInfo2018.columns:
    dfInfo2018[col] = dfInfo2018[col].astype(str)

dfInfo2018['text'] = dfInfo2018['Country'] + '<br>' + \
    '.Rank : ' + dfInfo2018['Happiness Rank']  + '<br>' + \
    '.Economy : ' + dfInfo2018['GDP per Capita'] + '<br>' + \
    '.Health : ' + dfInfo2018['Life Expectancy'] + '<br>' + \
    '.Freedom : ' + dfInfo2018['Freedom']  + '<br>' + \
    '.Government Corruption : ' + dfInfo2018['Government Corruption'] + '<br>' + \
    '.Generosity : ' + dfInfo2018['Generosity']

trace_2018=go.Choropleth(locations=dfInfo2018['Country'], # Spatial coordinates
                         z = dfInfo2018['Happiness Score'].astype(float), # Data to be color-coded
                         locationmode = 'country names', # set of locations match entries in `locations`
                         colorscale = 'RdBu',
                         autocolorscale=False,
                         text=dfInfo2018['text'],
                         colorbar_title = "Happiness")

dfInfo2019 = pd.DataFrame(data =df_2019)
for col in dfInfo2019.columns:
    dfInfo2019[col] = dfInfo2019[col].astype(str)

dfInfo2019['text'] = dfInfo2019['Country'] + '<br>' + \
    '.Rank : ' + dfInfo2019['Happiness Rank']  + '<br>' + \
    '.Economy : ' + dfInfo2019['GDP per Capita'] + '<br>' + \
    '.Health : ' + dfInfo2019['Life Expectancy'] + '<br>' + \
    '.Freedom : ' + dfInfo2019['Freedom']  + '<br>' + \
    '.Government Corruption : ' + dfInfo2019['Government Corruption'] + '<br>' + \
    '.Generosity : ' + dfInfo2019['Generosity']

trace_2019=go.Choropleth(locations=dfInfo2019['Country'], # Spatial coordinates
                         z = dfInfo2019['Happiness Score'].astype(float), # Data to be color-coded
                         locationmode = 'country names', # set of locations match entries in `locations`
                         colorscale = 'RdBu',
                         autocolorscale=False,
                         text=dfInfo2019['text'],
                         colorbar_title = "Happiness")
##################################################################### Plot Linha ######################################################################
avg = [(pd.to_numeric(df_2015['Happiness Score']).mean()), (pd.to_numeric(df_2016['Happiness Score']).mean()), (pd.to_numeric(df_2017['Happiness Score']).mean()), (pd.to_numeric(df_2018['Happiness Score']).mean()),(pd.to_numeric(df_2019['Happiness Score']).mean())]
layout=go.Layout(title='Happiness Score over the years')
fig2 = go.Figure(data=go.Scatter(x=["2015","2016","2017","2018","2019"], y=avg), layout=layout)
fig2.update_xaxes(nticks=5)
data = [trace_2015]
fig = go.Figure(data=data)

layout= go.Layout(title='Happiest Country')
figTOP = go.Figure(layout=layout)
# Constants
img_width = 500
img_height = 250
scale_factor = 0.5

# Add invisible scatter trace.
# This trace is added to help the autoresize logic work.
figTOP.add_trace(
    go.Scatter(
        x=[0, img_width * scale_factor],
        y=[0, img_height * scale_factor],
        mode="markers",
        marker_opacity=0
    )
)

# Configure axes
figTOP.update_xaxes(
    visible=False,
    range=[0, img_width * scale_factor]
)

figTOP.update_yaxes(
    visible=False,
    range=[0, img_height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
)

# Configure other layout
trace_TOP2015 = go.layout.Image(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="https://images-na.ssl-images-amazon.com/images/I/31sQ0fJmMmL._SY355_.jpg")

trace_TOP2016 = go.layout.Image(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="https://cdn.landfallnavigation.com/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/n/f/nf032_.gif")

trace_TOP2017 = go.layout.Image(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Flag_of_Norway.svg/2000px-Flag_of_Norway.svg.png")

trace_TOP2018 = go.layout.Image(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Flag_of_Finland.svg/1800px-Flag_of_Finland.svg.png")

######



######



##################################################################### Layout ######################################################################
app.layout = html.Div([
    html.Div(
        [
            html.H1('Happiness in the World'),
            html.H4('World overview'),
        ], id=('title'), style={"text-align": "center", "font-family": "Arial", "font-weight": "normal"}
    ),
    html.Div([
        html.Div([
            dcc.Slider(
                id='slider',
                min=2015,
                max=2019,
                marks={i: format(i) for i in range(2015, 2020)},
                value=2015,
            )], style={"width": "30%" , "float":"rigth", "margin-left":"5%"}
        ),
        html.Div(
            [
                dcc.Graph(id='avg_happy', figure=fig2,style={"width": "60%", "height": "300px", "float": "left", "margin-left": "40%"}),
                dcc.Graph(id='fig_Ranking', figure=figRankingGroups,
                          style={"width": "50%", "height": "10%", "float": "left", "margin-left": "50%"}),
                dcc.Graph(id='top1', figure=figTOP,
                          style={"width": "30%", "height": "300px", "float": "rigth", "margin-left": "5%", "margin-top": "5%"}),
                dcc.Graph(id='country_display', figure=fig,
                          style={"width": "50%", "height": "10%", "float": "rigth"}),
                dcc.Graph(id='scatter2', figure=figScatter2,
                          style={"width": "90%", "height": "10%", "float": "left", "margin-left": "5%"})]),
        html.Div(
            dcc.Dropdown(
                id='CountryDropdown',
                options=[{'label': i, 'value': i} for i in df_2015['Country'].unique()
                         ],
                value='Portugal'
            ), style={"width": "30%", "float": "rigth", "margin-left": "5%", "margin-top": "3%"}),
        html.Div(
            [
                dcc.Graph(id='scatter1', figure=figScatter1,
                          style={"width": "60%", "height": "300px", "float": "left", "margin-left": "40%"}),
                dcc.Graph(id='fig_Bar', figure=figBar,
                          style={"width": "60%", "height": "10%", "float": "left", "margin-left": "40%"}),
                dcc.Graph(id='fig_Corr', figure=figCorr,
                          style={"width": "40%", "height": "800px", "float": "rigth"}),
            ],
        ),
    ])
])


##################################################################### Callbacks ######################################################################
fig = go.Figure()
@app.callback([Output('country_display','figure'),Output('scatter2', 'figure'),Output('fig_Corr', 'figure'),Output('top1','figure')],[Input('slider','value')])
def update(year):
    if year == 2015:
        trace= trace_2015
        tracescatt = traceScatter2015
        corr = traceCorr2015
        traceTOP = trace_TOP2015
    elif year == 2016:
        trace = trace_2016
        tracescatt = traceScatter2016
        corr = traceCorr2016
        traceTOP = trace_TOP2016
    elif year == 2017:
        trace = trace_2017
        tracescatt = traceScatter2017
        corr = traceCorr2017
        traceTOP = trace_TOP2017
    elif year == 2018:
        trace = trace_2018
        tracescatt = traceScatter2018
        traceTOP = trace_TOP2018
        corr = traceCorr2018
    elif year == 2019:
        trace = trace_2019
        tracescatt = traceScatter2019
        corr = traceCorr2019
        traceTOP = trace_TOP2018

    data= [trace]
    dataScatt = [tracescatt]
    dataCorr_=[corr]
    layout=go.Layout(title='Happiness Score in '+str(year))
    fig = go.Figure(data=data,layout=layout)
    Scatterlayout = go.Layout(title='Government Trust Contribution in Happiness in '+str(year),
                              xaxis=dict(title='Happiness Score'),
                              yaxis=dict(title='Government Trust'))
    figScatter2 = go.Figure(data=dataScatt, layout= Scatterlayout)
    layout=go.Layout(title='Correlation Matrix in '+str(year))
    figCorr= go.Figure(data=dataCorr_, layout=layout)
    return fig,figScatter2,figCorr,figTOP.add_layout_image(traceTOP)



##### CLAUDIA 2

@app.callback(Output('fig_Ranking','figure'),[Input('slider','value')])
def update(year):
    if year == 2015:
        val = df_2015
    elif year == 2016:
        val = df_2016
    elif year == 2017:
        val = df_2017
    elif year == 2018:
        val = df_2018
    elif year == 2019:
        val = df_2019
    ############ Graph 2015

    df_total = pd.DataFrame(val.groupby('Region')['Country'].nunique()).rename(columns={'Country':'T'})

    #top and bottom countries
    df1 = val.iloc[:int(val.shape[0] * 0.25),
          [val.columns.get_loc("Country"), val.columns.get_loc("Region")]]
    df2 = val.iloc[int(val.shape[0] * 0.25):int(val.shape[0] * 0.5),
          [val.columns.get_loc("Country"), val.columns.get_loc("Region")]]
    df3 = val.iloc[int(val.shape[0] * 0.5):int(val.shape[0] * 0.75),
          [val.columns.get_loc("Country"), val.columns.get_loc("Region")]]
    df4 = val.iloc[int(val.shape[0] * 0.75):,
          [val.columns.get_loc("Country"), val.columns.get_loc("Region")]]

    df1_c = pd.DataFrame(df1.groupby('Region')['Country'].nunique()).rename(columns={'Country':'1'})
    df2_c = pd.DataFrame(df2.groupby('Region')['Country'].nunique()).rename(columns={'Country':'2'})
    df3_c = pd.DataFrame(df3.groupby('Region')['Country'].nunique()).rename(columns={'Country':'3'})
    df4_c = pd.DataFrame(df4.groupby('Region')['Country'].nunique()).rename(columns={'Country':'4'})

    df_region = pd.concat([df_total, df1_c, df2_c, df3_c, df4_c], axis=1, sort=False).fillna(0)

    df_cont = df_region.rename(index={'Australia and New Zealand':"Oceania",
                                       "Central and Eastern Europe":"Europe",
                                       "Eastern Asia":"Asia",
                                       "Latin America and Caribbean":"America",
                                       "Middle East and Northern Africa":"Africa",
                                       "North America":"America",
                                       "Southeastern Asia":"Asia",
                                       "Southern Asia":"Asia",
                                       "Sub-Saharan Africa":"Africa",
                                       "Western Europe": "Europe"})

    df_cont = df_cont.groupby(level=0).sum()

    #percentage table
    df_percentage = pd.DataFrame(round(df_cont['1']/df_cont['T'],2)*100, columns = ['df1_p'])
    df_percentage['df2_p']= round(df_cont['2']/df_cont['T'],2)*100
    df_percentage['df3_p']=round(df_cont['3']/df_cont['T'],2)*100
    df_percentage['df4_p']=round(df_cont['4']/df_cont['T'],2)*100

    #traces
    t1 = {
        "name": "Africa",
        "type": "bar",
        "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
        "y": df_percentage.iloc[0,:].values.tolist(),
        "xaxis": "x",
        "yaxis": "y"
    }

    t2 = {
        "name": "America",
        "type": "bar",
        "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
        "y": df_percentage.iloc[1,:].values.tolist(),
        "xaxis": "x",
        "yaxis": "y"
    }

    t3 = {
        "name": "Asia",
        "type": "bar",
        "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
        "y": df_percentage.iloc[2,:].values.tolist(),
        "xaxis": "x",
        "yaxis": "y"
    }

    t4 = {
        "name": "Europe",
        "type": "bar",
        "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
        "y": df_percentage.iloc[3,:].values.tolist(),
        "xaxis": "x",
        "yaxis": "y"
    }

    t5 = {
        "name": "Oceania",
        "type": "bar",
        "x": ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"],
        "y": df_percentage.iloc[4,:].values.tolist(),
        "xaxis": "x",
        "yaxis": "y"
    }

    dataRanking = [t1,t2,t3,t4,t5]

    layout = go.Layout(
        title = 'Percentages by continent by ranking groups in '+str(year),
        xaxis = dict(title='Happiness ranking groups'),
        yaxis = dict(title='Percentage'),
        barmode = "group"
    )

    figRankingGroups = go.Figure(data=dataRanking, layout=layout)

    return figRankingGroups


@app.callback(Output('scatter1','figure'),[Input('CountryDropdown','value')])
def updateCountry(value):
    print("{}".format(value))
    traceScatter = go.Scatter(x=["2015", "2016", "2017", "2018", "2019"],
                              y=[(
                                 pd.to_numeric(df_2015[df_2015['Country'] == "{}".format(value)]['Happiness Score']).values[0]),
                                 (
                                 pd.to_numeric(df_2016[df_2016['Country'] == "{}".format(value)]['Happiness Score']).values[0]),
                                 (
                                 pd.to_numeric(df_2017[df_2017['Country'] == "{}".format(value)]['Happiness Score']).values[0]),
                                 (pd.to_numeric(df_2018[df_2018['Country'] == "{}".format(value)]['Happiness Score'])).values[
                                     0],
                                 (pd.to_numeric(df_2019[df_2019['Country'] == "{}".format(value)]['Happiness Score']).values[
                                     0])],
                              name='Country Happiness Evolution')
    dataScatter = [traceScatter]
    layout=go.Layout(title='Happiness Score over the years in '+str(value))
    figScatter1 = go.Figure(data=dataScatter,layout=layout)
    figScatter1.update_xaxes(nticks=5)
    return figScatter1



if __name__=="__main__":
    app.run_server(debug=True)
