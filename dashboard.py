import time  # to simulate a real time data, time loop

import streamlit as st
import pandas as pd
import pyproj
import numpy as np
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import pydeck as pdk
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Transport Data Analysis Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


st.title("Trip Data Analysis")

data = pd.read_csv("https://raw.githubusercontent.com/zuocsfm/travel_data_dashboard/main/eqasim_legs.csv", sep=';')
transport_mode_list = data['mode'].unique().tolist()
departure_time_list = data['departure_time'].unique().tolist()
travel_time_list = data['travel_time'].unique().tolist()

# ------------------------------------------------------------------------
#  sidebar
# ------------------------------------------------------------------------

with st.sidebar:
    st.write("Data source: MATSim (https://www.matsim.org/)")
    st.title("Data selection")

    new_departure = st.slider(label="Select a departure time range (hour):",
                           min_value=6,
                           max_value=24,
                           value=(6, 20))

    new_departure = tuple((i*60*60) for i in new_departure)

    new_travel_time = st.slider(label='Select a travel duration time range (hour):',
                                   min_value=0,
                                   max_value=22,
                                   value=(0,22))
    new_travel_time = tuple((i*60*60) for i in new_travel_time)

    new_distance = st.slider(label='Select a routed distance range (km):',
                                   min_value=0,
                                   max_value=172,
                                   value=(0,172))
    new_distance = tuple((i * 1000) for i in new_travel_time)

    new_mode = st.multiselect("Choose transport mode:", transport_mode_list, transport_mode_list )

# filter data according to user selection
selected_subset = (data['departure_time'].between(*new_departure)) \
                  & (data['travel_time'].between(*new_travel_time)) & (data['mode'].isin(new_mode)\
                    & (data['routed_distance'].between(*new_distance)))

selected_subset = data[selected_subset]

# ------------------------------------------------------------------------
#  calculate the coordinates
# ------------------------------------------------------------------------

# convert the coordinates to latitude and longitude
proj = pyproj.Transformer.from_crs( 2154, 4326, always_xy=True)

# get the latitude and longitude
selected_subset['origin'] = selected_subset.apply(lambda row: proj.transform(row['origin_x'], row['origin_y']), axis=1)
selected_subset[['origin_lon', 'origin_lat']] = pd.DataFrame(selected_subset['origin'].tolist(), index=selected_subset.index)

# get the latitude and longitude
selected_subset['destination'] = selected_subset.apply(lambda row: proj.transform(row['destination_x'], row['destination_y']), axis=1)
selected_subset[['destination_lon', 'destination_lat']] = pd.DataFrame(selected_subset['destination'].tolist(), index=selected_subset.index)



# ------------------------------------------------------------------------
#  Summary - row 1
# ------------------------------------------------------------------------

summary1, summary2, summary3, summary4, summary5 = st.columns(5)

# display the statistics

trip_number = len((selected_subset['person_id'].astype(str) + "_" + selected_subset['person_trip_id'].astype(str)).unique())
summary1.metric("Number of trips", trip_number)

leg_number = len(selected_subset.index)
summary2.metric("Number of legs", leg_number)

person_number = len(selected_subset['person_id'].unique())
summary3.metric("Number of persons", person_number)

travel_time_average = selected_subset['travel_time'].mean()
summary4.metric("Average travel time (minutes)", (travel_time_average/60).round(2))

routed_distance_ave = selected_subset['routed_distance'].mean().round(2)
summary5.metric("Average routed distance (km)", (routed_distance_ave/1000).round(2))

style_metric_cards()

# ------------------------------------------------------------------------
#  Chart - row 2
# ------------------------------------------------------------------------

row2_1, row2_2, row2_3 = st.columns(3)

# ------------------------------------------------------------------------
#  Chart
# ------------------------------------------------------------------------

chart1, chart2, chart3, chart4 = st.columns(4)
travel_mode = selected_subset.groupby(['mode'])['mode'].count()
travel_mode = pd.DataFrame({'mode':travel_mode.index, 'number':travel_mode.values})

row2_3.write("Number of Trips by Travel Mode")
row2_3.bar_chart(travel_mode, x='mode', y='number')



# ------------------------------------------------------------------------
#  Map
# ------------------------------------------------------------------------

# draw the origins
# get the latitude and longitude
selected_subset['origin'] = selected_subset.apply(lambda row: proj.transform(row['origin_x'], row['origin_y']), axis=1)
selected_subset[['origin_lon', 'origin_lat']] = pd.DataFrame(selected_subset['origin'].tolist(), index=selected_subset.index)

# chart1.write("The location of origins")
# chart1.map(selected_subset, latitude='origin_lat', longitude='origin_lon', size = 10, color='#00445f')

# draw the destinations
# get the latitude and longitude
selected_subset['destination'] = selected_subset.apply(lambda row: proj.transform(row['destination_x'], row['destination_y']), axis=1)
selected_subset[['destination_lon', 'destination_lat']] = pd.DataFrame(selected_subset['destination'].tolist(), index=selected_subset.index)

# chart2.write("The location of destinations")
# chart2.map(selected_subset, latitude='destination_lat', longitude='destination_lon')

# ------------------------------------------------------------------------
#  Flow map
# ------------------------------------------------------------------------

# draw the flow map
GREEN_RGB = [98, 115, 19, 80]
RED_RGB = [183, 53, 45, 80]

row2_1.write("The origins and destinations")
row2_1.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=42,
        longitude=9.1,
        zoom=8,
        pitch=170,
    ),
    layers=[
        pdk.Layer(
           "ArcLayer",
            data=selected_subset,
            get_width="S000 * 2",
            get_source_position=["origin_lon", "origin_lat"],
            get_target_position=["destination_lon", "destination_lat"],
            get_tilt=15,
            get_source_color=RED_RGB,
            get_target_color=GREEN_RGB,
            pickable=True,
            auto_highlight=True,
        ),
    ],
    tooltip={
        'html': '<b>Person id:</b> {person_id}<br><b>Trip id:</b> {person_trip_id}<br><b>Leg index:</b> {leg_index}',
        'style': {
            'color': 'white'
        }
    }
))

# ------------------------------------------------------------------------
#  map the stops
# ------------------------------------------------------------------------
#

# calculate breaks

selected_subset['arrival_time'] = selected_subset['departure_time'] + selected_subset['travel_time']
max_trip = max(selected_subset['leg_index'].unique().tolist())

df_stops = pd.DataFrame(columns=['person_id', 'stop_index', 'lat', 'lon', 'end_time', 'start_time', 'duration', 'mode'])

for i in range(1, max_trip+1):
    # find arrival and departure trip pairs
    df_arrival = selected_subset.loc[selected_subset['leg_index'] == i]
    df_departure = selected_subset.loc[selected_subset['leg_index'] == (i-1)]

    arrival_person = df_arrival['person_id'].unique().tolist()
    departure_person = df_departure['person_id'].unique().tolist()

    # calculate the person made stops
    common_person = set(arrival_person) & set(departure_person)

    for p in common_person:
        new_stop = {}
        new_stop['person_id'] = p
        new_stop['stop_index'] = i
        new_stop['lat'] = selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == i)]['origin_lat']
        new_stop['lon'] = selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == i)]['origin_lon']
        new_stop['end_time'] = int(selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == i)]['departure_time'].tolist()[0])
        new_stop['start_time'] = int(selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == (i - 1))]['arrival_time'].tolist()[0])
        new_stop['duration'] = new_stop['end_time'] - new_stop['start_time']
        new_stop['mode'] = selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == i)]['mode']

        df_new_stop = pd.DataFrame.from_dict(new_stop)
        df_stops = pd.concat([df_stops, df_new_stop], ignore_index=True)

row2_2.write("The stops in trips")
row2_2.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=41.9,
        longitude=9.1,
        zoom=8,
        pitch=170,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df_stops,
           get_position='[lon, lat]',
           radius=500,
           elevation_scale=4,
           elevation_range=[0, 5000],
           pickable=True,
           extruded=True,
        ),
    ],
    tooltip={
            'html': '<b>Number of stops:</b> {elevationValue}',
            'style': {
                'color': 'white'
            }
        }
))

# ------------------------------------------------------------------------
#  Show the Raw data
# ------------------------------------------------------------------------
# delete the intermediate columns
selected_subset.drop(['origin','destination', 'origin_lon', 'origin_lat', 'destination_lat', 'destination_lon'], axis='columns', inplace=True)

st.write("Filtered dataset")

st.dataframe(selected_subset, width=2000)


# ------------------------------------------------------------------------
#  data download
# ------------------------------------------------------------------------

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=';').encode('utf-8')

csv = convert_df(selected_subset)

with st.sidebar:
    st.write("\n")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_eqasim_data.csv',
        mime='text/csv',
    )







