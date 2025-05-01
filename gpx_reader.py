import xml.etree.ElementTree as ET
import pandas as pd

def gpx_to_dataframe(gpx_path: str) -> pd.DataFrame:
    """
    Parse a .gpx file and return a DataFrame with columns:
      - time:   UTC timestamp of each track point
      - lat:    latitude
      - long:   longitude
    """
    # define the GPX namespace
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    # parse XML
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    
    # collect rows
    rows = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        time_el = trkpt.find('gpx:time', ns)
        timestamp = pd.to_datetime(time_el.text) if time_el is not None else pd.NaT
        rows.append({'time': timestamp, 'lat': lat, 'long': lon})
    
    # build DataFrame
    df = pd.DataFrame(rows)
    return df


import matplotlib.pyplot as plt
from matplotlib import dates as mdates

def plot_track(df):
    """
    Scatter-plot longitude vs. latitude, with marker colors mapped to time.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'time': datetime64[ns]
          - 'lat' : float
          - 'long': float
    """
    # convert timestamps to matplotlib’s numeric format
    times_num = mdates.date2num(df['time'])
    
    # create figure and axis
    fig, ax = plt.subplots()
    
    # scatter: x=long, y=lat, color=time
    sc = ax.scatter(
        df['long'], 
        df['lat'], 
        c=times_num, 
        cmap='viridis',
        marker='o'
    )
    
    # add a colorbar with formatted datetime ticks
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    )
    cbar.set_label('Time')
    
    # labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GPS Track (colored by time)')
    
    # improve date label formatting
    fig.autofmt_xdate()
    
    plt.show()

import numpy as np

def add_distance_column(df):
    """
    Given a DataFrame with 'lat' and 'long' in decimal degrees,
    adds a 'distance' column (meters) giving the distance
    from the previous point. The first row is 0.
    """
    # Earth radius in meters
    R = 6_371_000

    # convert degrees to radians
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['long'])

    # previous point (for first row, prev = current → Δ=0)
    lat_prev = lat_rad.shift().fillna(lat_rad)
    lon_prev = lon_rad.shift().fillna(lon_rad)

    # deltas
    dlat = lat_rad - lat_prev
    dlon = lon_rad - lon_prev

    # haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # distance column
    df['distance'] = R * c

    return df

# Example usage:
# df = gpx_to_dataframe('track.gpx')
# df = add_distance_column(df)
# print(df.head())


# Example usage:
df = gpx_to_dataframe('Apr_28,_2025_11_35_56_AM.gpx')
df = add_distance_column(df)
#take just the last 1000 points
#df = df.iloc[-120:]
plot_track(df)
print(df.head())

# now plot the distance
plt.plot(df['time'], df['distance'])
plt.xlabel('Time')
plt.ylabel('Distance (m)')
plt.title('Distance over Time')
plt.show()

# plot the cumulative distance
plt.plot(df['time'], df['distance'].cumsum())
plt.xlabel('Time')
plt.ylabel('Cumulative Distance (m)')
plt.title('Cumulative Distance over Time')
plt.show()
