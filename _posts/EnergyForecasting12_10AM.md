# Energy Forecasting - Time-Series Modeling

## Logan Heine


```python
import numpy as np
import pandas as pd
import datetime as dt

# plotting
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from statsmodels.api import tsa # time series analysis
import statsmodels.api as sm
#from statsmodels.tsa.seasonal import MSTL
#Need statsmodels 0.14.0 for this one
from statsmodels.tsa.seasonal import DecomposeResult
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GRU, BatchNormalization, Flatten
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing.text import Tokenizer


import warnings
warnings.filterwarnings('ignore')
```


```python
sm.__version__
```




    '0.13.5'



## Exploratory Data Analysis


```python
df_e = pd.read_csv('data/energy_dataset.csv')
```


```python
df_e.head(10)
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
      <th>time</th>
      <th>generation biomass</th>
      <th>generation fossil brown coal/lignite</th>
      <th>generation fossil coal-derived gas</th>
      <th>generation fossil gas</th>
      <th>generation fossil hard coal</th>
      <th>generation fossil oil</th>
      <th>generation fossil oil shale</th>
      <th>generation fossil peat</th>
      <th>generation geothermal</th>
      <th>...</th>
      <th>generation waste</th>
      <th>generation wind offshore</th>
      <th>generation wind onshore</th>
      <th>forecast solar day ahead</th>
      <th>forecast wind offshore eday ahead</th>
      <th>forecast wind onshore day ahead</th>
      <th>total load forecast</th>
      <th>total load actual</th>
      <th>price day ahead</th>
      <th>price actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01 00:00:00+01:00</td>
      <td>447.0</td>
      <td>329.0</td>
      <td>0.0</td>
      <td>4844.0</td>
      <td>4821.0</td>
      <td>162.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>196.0</td>
      <td>0.0</td>
      <td>6378.0</td>
      <td>17.0</td>
      <td>NaN</td>
      <td>6436.0</td>
      <td>26118.0</td>
      <td>25385.0</td>
      <td>50.10</td>
      <td>65.41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01 01:00:00+01:00</td>
      <td>449.0</td>
      <td>328.0</td>
      <td>0.0</td>
      <td>5196.0</td>
      <td>4755.0</td>
      <td>158.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>5890.0</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>5856.0</td>
      <td>24934.0</td>
      <td>24382.0</td>
      <td>48.10</td>
      <td>64.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01 02:00:00+01:00</td>
      <td>448.0</td>
      <td>323.0</td>
      <td>0.0</td>
      <td>4857.0</td>
      <td>4581.0</td>
      <td>157.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>196.0</td>
      <td>0.0</td>
      <td>5461.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>5454.0</td>
      <td>23515.0</td>
      <td>22734.0</td>
      <td>47.33</td>
      <td>64.48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01 03:00:00+01:00</td>
      <td>438.0</td>
      <td>254.0</td>
      <td>0.0</td>
      <td>4314.0</td>
      <td>4131.0</td>
      <td>160.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>191.0</td>
      <td>0.0</td>
      <td>5238.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>5151.0</td>
      <td>22642.0</td>
      <td>21286.0</td>
      <td>42.27</td>
      <td>59.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01 04:00:00+01:00</td>
      <td>428.0</td>
      <td>187.0</td>
      <td>0.0</td>
      <td>4130.0</td>
      <td>3840.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>4935.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>4861.0</td>
      <td>21785.0</td>
      <td>20264.0</td>
      <td>38.41</td>
      <td>56.04</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-01-01 05:00:00+01:00</td>
      <td>410.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>4038.0</td>
      <td>3590.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>188.0</td>
      <td>0.0</td>
      <td>4618.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4617.0</td>
      <td>21441.0</td>
      <td>19905.0</td>
      <td>35.72</td>
      <td>53.63</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-01-01 06:00:00+01:00</td>
      <td>401.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>4040.0</td>
      <td>3368.0</td>
      <td>158.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>186.0</td>
      <td>0.0</td>
      <td>4397.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4276.0</td>
      <td>21285.0</td>
      <td>20010.0</td>
      <td>35.13</td>
      <td>51.73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-01-01 07:00:00+01:00</td>
      <td>408.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>4030.0</td>
      <td>3208.0</td>
      <td>160.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>189.0</td>
      <td>0.0</td>
      <td>3992.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>3994.0</td>
      <td>21545.0</td>
      <td>20377.0</td>
      <td>36.22</td>
      <td>51.43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-01-01 08:00:00+01:00</td>
      <td>413.0</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>4052.0</td>
      <td>3335.0</td>
      <td>161.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>198.0</td>
      <td>0.0</td>
      <td>3629.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>3602.0</td>
      <td>21443.0</td>
      <td>20094.0</td>
      <td>32.40</td>
      <td>48.98</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-01-01 09:00:00+01:00</td>
      <td>419.0</td>
      <td>177.0</td>
      <td>0.0</td>
      <td>4137.0</td>
      <td>3437.0</td>
      <td>163.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>198.0</td>
      <td>0.0</td>
      <td>3073.0</td>
      <td>784.0</td>
      <td>NaN</td>
      <td>3212.0</td>
      <td>21560.0</td>
      <td>20637.0</td>
      <td>36.60</td>
      <td>54.20</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 29 columns</p>
</div>




```python
df_e.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 35064 entries, 0 to 35063
    Data columns (total 29 columns):
     #   Column                                       Non-Null Count  Dtype  
    ---  ------                                       --------------  -----  
     0   time                                         35064 non-null  object 
     1   generation biomass                           35045 non-null  float64
     2   generation fossil brown coal/lignite         35046 non-null  float64
     3   generation fossil coal-derived gas           35046 non-null  float64
     4   generation fossil gas                        35046 non-null  float64
     5   generation fossil hard coal                  35046 non-null  float64
     6   generation fossil oil                        35045 non-null  float64
     7   generation fossil oil shale                  35046 non-null  float64
     8   generation fossil peat                       35046 non-null  float64
     9   generation geothermal                        35046 non-null  float64
     10  generation hydro pumped storage aggregated   0 non-null      float64
     11  generation hydro pumped storage consumption  35045 non-null  float64
     12  generation hydro run-of-river and poundage   35045 non-null  float64
     13  generation hydro water reservoir             35046 non-null  float64
     14  generation marine                            35045 non-null  float64
     15  generation nuclear                           35047 non-null  float64
     16  generation other                             35046 non-null  float64
     17  generation other renewable                   35046 non-null  float64
     18  generation solar                             35046 non-null  float64
     19  generation waste                             35045 non-null  float64
     20  generation wind offshore                     35046 non-null  float64
     21  generation wind onshore                      35046 non-null  float64
     22  forecast solar day ahead                     35064 non-null  float64
     23  forecast wind offshore eday ahead            0 non-null      float64
     24  forecast wind onshore day ahead              35064 non-null  float64
     25  total load forecast                          35064 non-null  float64
     26  total load actual                            35028 non-null  float64
     27  price day ahead                              35064 non-null  float64
     28  price actual                                 35064 non-null  float64
    dtypes: float64(28), object(1)
    memory usage: 7.8+ MB
    


```python
df_e.describe()
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
      <th>generation biomass</th>
      <th>generation fossil brown coal/lignite</th>
      <th>generation fossil coal-derived gas</th>
      <th>generation fossil gas</th>
      <th>generation fossil hard coal</th>
      <th>generation fossil oil</th>
      <th>generation fossil oil shale</th>
      <th>generation fossil peat</th>
      <th>generation geothermal</th>
      <th>generation hydro pumped storage aggregated</th>
      <th>...</th>
      <th>generation waste</th>
      <th>generation wind offshore</th>
      <th>generation wind onshore</th>
      <th>forecast solar day ahead</th>
      <th>forecast wind offshore eday ahead</th>
      <th>forecast wind onshore day ahead</th>
      <th>total load forecast</th>
      <th>total load actual</th>
      <th>price day ahead</th>
      <th>price actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>35045.000000</td>
      <td>35046.000000</td>
      <td>35046.0</td>
      <td>35046.000000</td>
      <td>35046.000000</td>
      <td>35045.000000</td>
      <td>35046.0</td>
      <td>35046.0</td>
      <td>35046.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>35045.000000</td>
      <td>35046.0</td>
      <td>35046.000000</td>
      <td>35064.000000</td>
      <td>0.0</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
      <td>35028.000000</td>
      <td>35064.000000</td>
      <td>35064.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>383.513540</td>
      <td>448.059208</td>
      <td>0.0</td>
      <td>5622.737488</td>
      <td>4256.065742</td>
      <td>298.319789</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>269.452133</td>
      <td>0.0</td>
      <td>5464.479769</td>
      <td>1439.066735</td>
      <td>NaN</td>
      <td>5471.216689</td>
      <td>28712.129962</td>
      <td>28696.939905</td>
      <td>49.874341</td>
      <td>57.884023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.353943</td>
      <td>354.568590</td>
      <td>0.0</td>
      <td>2201.830478</td>
      <td>1961.601013</td>
      <td>52.520673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>50.195536</td>
      <td>0.0</td>
      <td>3213.691587</td>
      <td>1677.703355</td>
      <td>NaN</td>
      <td>3176.312853</td>
      <td>4594.100854</td>
      <td>4574.987950</td>
      <td>14.618900</td>
      <td>14.204083</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>237.000000</td>
      <td>18105.000000</td>
      <td>18041.000000</td>
      <td>2.060000</td>
      <td>9.330000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>333.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4126.000000</td>
      <td>2527.000000</td>
      <td>263.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>240.000000</td>
      <td>0.0</td>
      <td>2933.000000</td>
      <td>69.000000</td>
      <td>NaN</td>
      <td>2979.000000</td>
      <td>24793.750000</td>
      <td>24807.750000</td>
      <td>41.490000</td>
      <td>49.347500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>367.000000</td>
      <td>509.000000</td>
      <td>0.0</td>
      <td>4969.000000</td>
      <td>4474.000000</td>
      <td>300.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>279.000000</td>
      <td>0.0</td>
      <td>4849.000000</td>
      <td>576.000000</td>
      <td>NaN</td>
      <td>4855.000000</td>
      <td>28906.000000</td>
      <td>28901.000000</td>
      <td>50.520000</td>
      <td>58.020000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>433.000000</td>
      <td>757.000000</td>
      <td>0.0</td>
      <td>6429.000000</td>
      <td>5838.750000</td>
      <td>330.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>310.000000</td>
      <td>0.0</td>
      <td>7398.000000</td>
      <td>2636.000000</td>
      <td>NaN</td>
      <td>7353.000000</td>
      <td>32263.250000</td>
      <td>32192.000000</td>
      <td>60.530000</td>
      <td>68.010000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>592.000000</td>
      <td>999.000000</td>
      <td>0.0</td>
      <td>20034.000000</td>
      <td>8359.000000</td>
      <td>449.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>357.000000</td>
      <td>0.0</td>
      <td>17436.000000</td>
      <td>5836.000000</td>
      <td>NaN</td>
      <td>17430.000000</td>
      <td>41390.000000</td>
      <td>41015.000000</td>
      <td>101.990000</td>
      <td>116.800000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>




```python
df_w = pd.read_csv('data/weather_features.csv')
df_w.head()
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
      <th>dt_iso</th>
      <th>city_name</th>
      <th>temp</th>
      <th>temp_min</th>
      <th>temp_max</th>
      <th>pressure</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>wind_deg</th>
      <th>rain_1h</th>
      <th>rain_3h</th>
      <th>snow_3h</th>
      <th>clouds_all</th>
      <th>weather_id</th>
      <th>weather_main</th>
      <th>weather_description</th>
      <th>weather_icon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-01-01 00:00:00+01:00</td>
      <td>Valencia</td>
      <td>270.475</td>
      <td>270.475</td>
      <td>270.475</td>
      <td>1001</td>
      <td>77</td>
      <td>1</td>
      <td>62</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>800</td>
      <td>clear</td>
      <td>sky is clear</td>
      <td>01n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01 01:00:00+01:00</td>
      <td>Valencia</td>
      <td>270.475</td>
      <td>270.475</td>
      <td>270.475</td>
      <td>1001</td>
      <td>77</td>
      <td>1</td>
      <td>62</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>800</td>
      <td>clear</td>
      <td>sky is clear</td>
      <td>01n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01 02:00:00+01:00</td>
      <td>Valencia</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>1002</td>
      <td>78</td>
      <td>0</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>800</td>
      <td>clear</td>
      <td>sky is clear</td>
      <td>01n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01 03:00:00+01:00</td>
      <td>Valencia</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>1002</td>
      <td>78</td>
      <td>0</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>800</td>
      <td>clear</td>
      <td>sky is clear</td>
      <td>01n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01 04:00:00+01:00</td>
      <td>Valencia</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>269.686</td>
      <td>1002</td>
      <td>78</td>
      <td>0</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>800</td>
      <td>clear</td>
      <td>sky is clear</td>
      <td>01n</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_w.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178396 entries, 0 to 178395
    Data columns (total 17 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   dt_iso               178396 non-null  object 
     1   city_name            178396 non-null  object 
     2   temp                 178396 non-null  float64
     3   temp_min             178396 non-null  float64
     4   temp_max             178396 non-null  float64
     5   pressure             178396 non-null  int64  
     6   humidity             178396 non-null  int64  
     7   wind_speed           178396 non-null  int64  
     8   wind_deg             178396 non-null  int64  
     9   rain_1h              178396 non-null  float64
     10  rain_3h              178396 non-null  float64
     11  snow_3h              178396 non-null  float64
     12  clouds_all           178396 non-null  int64  
     13  weather_id           178396 non-null  int64  
     14  weather_main         178396 non-null  object 
     15  weather_description  178396 non-null  object 
     16  weather_icon         178396 non-null  object 
    dtypes: float64(6), int64(6), object(5)
    memory usage: 23.1+ MB
    

## Visualizations at Different Scales


```python
df_e['time'] = pd.to_datetime(df_e['time'], utc=True)
df_w['dt_iso'] = pd.to_datetime(df_w['dt_iso'])
```


```python
df_e2 = df_e.set_index('time')
```


```python
df_e2['total load actual'].plot(figsize=(20, 10))
plt.title('Four Years of Electricity Demand')
plt.show()
```


    
![png](output_13_0.png)
    



```python
df_e2['total load actual'][5000:9600].plot(figsize=(20, 10))
plt.title('Six Months of Electricity Demand')
plt.show()
```


    
![png](output_14_0.png)
    



```python
df_e2['total load actual'][0:750].plot(figsize=(20, 10))
plt.title('One Month of Electricity Demand')
plt.show()
```


    
![png](output_15_0.png)
    



```python
df_e2['total load actual'][3289:3460].plot(figsize=(20, 10))
plt.title('One Week of Electricity Demand (Monday through Sunday)')
plt.show()
```


    
![png](output_16_0.png)
    



```python
df_e2['total load actual'][3289:3314].plot(figsize=(20, 10))
plt.title('24 Hours of Electricity Demand (Monday)')
plt.show()
```


    
![png](output_17_0.png)
    


## Data Cleaning


```python
first_day = df_e2.index.min()
last_day = df_e2.index.max()

# pandas `Timestamp` objects
first_day, last_day
```




    (Timestamp('2014-12-31 23:00:00+0000', tz='UTC'),
     Timestamp('2018-12-31 22:00:00+0000', tz='UTC'))




```python
# Looks like there are no missing timestamps- we have every hour for the entire length of time
full_range = pd.date_range(start=first_day, end=last_day, freq="H")
full_range.difference(df_e2.index)
```




    DatetimeIndex([], dtype='datetime64[ns, UTC]', freq='H')




```python
df_e2 = df_e2.drop('generation hydro pumped storage aggregated', axis=1)
```


```python
df_e2 = df_e2.drop('forecast wind offshore eday ahead', axis=1)
```


```python
df_e2.isna().sum()
```




    generation biomass                             19
    generation fossil brown coal/lignite           18
    generation fossil coal-derived gas             18
    generation fossil gas                          18
    generation fossil hard coal                    18
    generation fossil oil                          19
    generation fossil oil shale                    18
    generation fossil peat                         18
    generation geothermal                          18
    generation hydro pumped storage consumption    19
    generation hydro run-of-river and poundage     19
    generation hydro water reservoir               18
    generation marine                              19
    generation nuclear                             17
    generation other                               18
    generation other renewable                     18
    generation solar                               18
    generation waste                               19
    generation wind offshore                       18
    generation wind onshore                        18
    forecast solar day ahead                        0
    forecast wind onshore day ahead                 0
    total load forecast                             0
    total load actual                              36
    price day ahead                                 0
    price actual                                    0
    dtype: int64




```python
# Looks like we are missisng at least some data in 47 rows
df_e2[df_e2.isna().any(axis=1)]
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
      <th>generation biomass</th>
      <th>generation fossil brown coal/lignite</th>
      <th>generation fossil coal-derived gas</th>
      <th>generation fossil gas</th>
      <th>generation fossil hard coal</th>
      <th>generation fossil oil</th>
      <th>generation fossil oil shale</th>
      <th>generation fossil peat</th>
      <th>generation geothermal</th>
      <th>generation hydro pumped storage consumption</th>
      <th>...</th>
      <th>generation solar</th>
      <th>generation waste</th>
      <th>generation wind offshore</th>
      <th>generation wind onshore</th>
      <th>forecast solar day ahead</th>
      <th>forecast wind onshore day ahead</th>
      <th>total load forecast</th>
      <th>total load actual</th>
      <th>price day ahead</th>
      <th>price actual</th>
    </tr>
    <tr>
      <th>time</th>
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
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-05 02:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>546.0</td>
      <td>8233.0</td>
      <td>21912.0</td>
      <td>21182.0</td>
      <td>35.20</td>
      <td>59.68</td>
    </tr>
    <tr>
      <th>2015-01-05 11:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3932.0</td>
      <td>9258.0</td>
      <td>23209.0</td>
      <td>NaN</td>
      <td>35.50</td>
      <td>79.14</td>
    </tr>
    <tr>
      <th>2015-01-05 12:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4236.0</td>
      <td>9156.0</td>
      <td>23725.0</td>
      <td>NaN</td>
      <td>36.80</td>
      <td>73.95</td>
    </tr>
    <tr>
      <th>2015-01-05 13:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4215.0</td>
      <td>9072.0</td>
      <td>23614.0</td>
      <td>NaN</td>
      <td>32.50</td>
      <td>71.93</td>
    </tr>
    <tr>
      <th>2015-01-05 14:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4050.0</td>
      <td>8779.0</td>
      <td>22381.0</td>
      <td>NaN</td>
      <td>30.00</td>
      <td>71.50</td>
    </tr>
    <tr>
      <th>2015-01-05 15:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3728.0</td>
      <td>8426.0</td>
      <td>21371.0</td>
      <td>NaN</td>
      <td>30.00</td>
      <td>71.85</td>
    </tr>
    <tr>
      <th>2015-01-05 16:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3175.0</td>
      <td>7946.0</td>
      <td>20760.0</td>
      <td>NaN</td>
      <td>30.60</td>
      <td>80.53</td>
    </tr>
    <tr>
      <th>2015-01-19 18:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>6434.0</td>
      <td>38642.0</td>
      <td>39304.0</td>
      <td>70.01</td>
      <td>88.95</td>
    </tr>
    <tr>
      <th>2015-01-19 19:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.0</td>
      <td>6907.0</td>
      <td>38758.0</td>
      <td>39262.0</td>
      <td>69.00</td>
      <td>87.94</td>
    </tr>
    <tr>
      <th>2015-01-27 18:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>182.0</td>
      <td>9807.0</td>
      <td>38968.0</td>
      <td>38335.0</td>
      <td>66.00</td>
      <td>83.97</td>
    </tr>
    <tr>
      <th>2015-01-28 12:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4136.0</td>
      <td>6223.0</td>
      <td>36239.0</td>
      <td>NaN</td>
      <td>65.00</td>
      <td>77.62</td>
    </tr>
    <tr>
      <th>2015-02-01 06:00:00+00:00</th>
      <td>449.0</td>
      <td>312.0</td>
      <td>0.0</td>
      <td>4765.0</td>
      <td>5269.0</td>
      <td>222.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>480.0</td>
      <td>...</td>
      <td>48.0</td>
      <td>208.0</td>
      <td>0.0</td>
      <td>3289.0</td>
      <td>18.0</td>
      <td>3141.0</td>
      <td>24379.0</td>
      <td>NaN</td>
      <td>56.10</td>
      <td>16.98</td>
    </tr>
    <tr>
      <th>2015-02-01 07:00:00+00:00</th>
      <td>453.0</td>
      <td>312.0</td>
      <td>0.0</td>
      <td>4938.0</td>
      <td>5652.0</td>
      <td>288.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>73.0</td>
      <td>207.0</td>
      <td>0.0</td>
      <td>3102.0</td>
      <td>63.0</td>
      <td>3165.0</td>
      <td>27389.0</td>
      <td>NaN</td>
      <td>57.69</td>
      <td>19.56</td>
    </tr>
    <tr>
      <th>2015-02-01 08:00:00+00:00</th>
      <td>452.0</td>
      <td>302.0</td>
      <td>0.0</td>
      <td>4997.0</td>
      <td>5770.0</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>809.0</td>
      <td>204.0</td>
      <td>0.0</td>
      <td>2838.0</td>
      <td>691.0</td>
      <td>2907.0</td>
      <td>30619.0</td>
      <td>NaN</td>
      <td>60.01</td>
      <td>23.13</td>
    </tr>
    <tr>
      <th>2015-02-01 11:00:00+00:00</th>
      <td>405.0</td>
      <td>317.0</td>
      <td>0.0</td>
      <td>5247.0</td>
      <td>6008.0</td>
      <td>333.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3817.0</td>
      <td>200.0</td>
      <td>0.0</td>
      <td>1413.0</td>
      <td>3568.0</td>
      <td>1486.0</td>
      <td>31357.0</td>
      <td>NaN</td>
      <td>59.97</td>
      <td>22.51</td>
    </tr>
    <tr>
      <th>2015-02-01 12:00:00+00:00</th>
      <td>402.0</td>
      <td>317.0</td>
      <td>0.0</td>
      <td>5449.0</td>
      <td>6005.0</td>
      <td>318.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3836.0</td>
      <td>193.0</td>
      <td>0.0</td>
      <td>1347.0</td>
      <td>3572.0</td>
      <td>1267.0</td>
      <td>31338.0</td>
      <td>NaN</td>
      <td>59.69</td>
      <td>23.44</td>
    </tr>
    <tr>
      <th>2015-02-01 13:00:00+00:00</th>
      <td>400.0</td>
      <td>317.0</td>
      <td>0.0</td>
      <td>5266.0</td>
      <td>5995.0</td>
      <td>327.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3701.0</td>
      <td>199.0</td>
      <td>0.0</td>
      <td>1345.0</td>
      <td>3474.0</td>
      <td>1363.0</td>
      <td>30874.0</td>
      <td>NaN</td>
      <td>58.69</td>
      <td>24.10</td>
    </tr>
    <tr>
      <th>2015-02-01 14:00:00+00:00</th>
      <td>393.0</td>
      <td>321.0</td>
      <td>0.0</td>
      <td>5209.0</td>
      <td>5939.0</td>
      <td>345.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>3475.0</td>
      <td>204.0</td>
      <td>0.0</td>
      <td>1487.0</td>
      <td>2935.0</td>
      <td>1426.0</td>
      <td>30124.0</td>
      <td>NaN</td>
      <td>58.13</td>
      <td>21.12</td>
    </tr>
    <tr>
      <th>2015-02-01 15:00:00+00:00</th>
      <td>413.0</td>
      <td>325.0</td>
      <td>0.0</td>
      <td>5642.0</td>
      <td>6000.0</td>
      <td>345.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2742.0</td>
      <td>203.0</td>
      <td>0.0</td>
      <td>1648.0</td>
      <td>2383.0</td>
      <td>1678.0</td>
      <td>29714.0</td>
      <td>NaN</td>
      <td>59.00</td>
      <td>21.73</td>
    </tr>
    <tr>
      <th>2015-02-01 16:00:00+00:00</th>
      <td>465.0</td>
      <td>321.0</td>
      <td>0.0</td>
      <td>6127.0</td>
      <td>5912.0</td>
      <td>346.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1281.0</td>
      <td>207.0</td>
      <td>0.0</td>
      <td>1857.0</td>
      <td>1211.0</td>
      <td>1807.0</td>
      <td>29801.0</td>
      <td>NaN</td>
      <td>59.69</td>
      <td>25.93</td>
    </tr>
    <tr>
      <th>2015-02-01 17:00:00+00:00</th>
      <td>482.0</td>
      <td>326.0</td>
      <td>0.0</td>
      <td>7386.0</td>
      <td>6002.0</td>
      <td>340.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>328.0</td>
      <td>208.0</td>
      <td>0.0</td>
      <td>1864.0</td>
      <td>392.0</td>
      <td>2050.0</td>
      <td>32257.0</td>
      <td>NaN</td>
      <td>63.76</td>
      <td>54.13</td>
    </tr>
    <tr>
      <th>2015-02-01 18:00:00+00:00</th>
      <td>474.0</td>
      <td>326.0</td>
      <td>0.0</td>
      <td>7963.0</td>
      <td>6026.0</td>
      <td>343.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>161.0</td>
      <td>207.0</td>
      <td>0.0</td>
      <td>1813.0</td>
      <td>98.0</td>
      <td>1933.0</td>
      <td>33183.0</td>
      <td>NaN</td>
      <td>65.01</td>
      <td>68.53</td>
    </tr>
    <tr>
      <th>2015-04-05 01:00:00+00:00</th>
      <td>371.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5015.0</td>
      <td>3248.0</td>
      <td>257.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>799.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>3153.0</td>
      <td>55.0</td>
      <td>3026.0</td>
      <td>20016.0</td>
      <td>NaN</td>
      <td>42.55</td>
      <td>29.04</td>
    </tr>
    <tr>
      <th>2015-04-16 07:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>952.0</td>
      <td>3377.0</td>
      <td>31001.0</td>
      <td>NaN</td>
      <td>56.54</td>
      <td>67.55</td>
    </tr>
    <tr>
      <th>2015-04-20 06:00:00+00:00</th>
      <td>424.0</td>
      <td>642.0</td>
      <td>0.0</td>
      <td>5614.0</td>
      <td>5784.0</td>
      <td>369.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>636.0</td>
      <td>147.0</td>
      <td>0.0</td>
      <td>797.0</td>
      <td>692.0</td>
      <td>830.0</td>
      <td>29287.0</td>
      <td>NaN</td>
      <td>62.00</td>
      <td>72.92</td>
    </tr>
    <tr>
      <th>2015-04-23 19:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>327.0</td>
      <td>1058.0</td>
      <td>31421.0</td>
      <td>NaN</td>
      <td>69.49</td>
      <td>82.57</td>
    </tr>
    <tr>
      <th>2015-05-02 08:00:00+00:00</th>
      <td>497.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5502.0</td>
      <td>5677.0</td>
      <td>375.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2535.0</td>
      <td>205.0</td>
      <td>0.0</td>
      <td>10903.0</td>
      <td>2277.0</td>
      <td>11069.0</td>
      <td>39644.0</td>
      <td>NaN</td>
      <td>58.49</td>
      <td>59.09</td>
    </tr>
    <tr>
      <th>2015-05-29 01:00:00+00:00</th>
      <td>569.0</td>
      <td>756.0</td>
      <td>0.0</td>
      <td>4239.0</td>
      <td>4635.0</td>
      <td>365.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>755.0</td>
      <td>...</td>
      <td>662.0</td>
      <td>201.0</td>
      <td>0.0</td>
      <td>6503.0</td>
      <td>554.0</td>
      <td>6355.0</td>
      <td>23132.0</td>
      <td>NaN</td>
      <td>45.93</td>
      <td>55.07</td>
    </tr>
    <tr>
      <th>2015-06-15 07:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1552.0</td>
      <td>2718.0</td>
      <td>29899.0</td>
      <td>30047.0</td>
      <td>62.48</td>
      <td>73.82</td>
    </tr>
    <tr>
      <th>2015-10-02 06:00:00+00:00</th>
      <td>483.0</td>
      <td>961.0</td>
      <td>0.0</td>
      <td>6545.0</td>
      <td>8250.0</td>
      <td>385.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>140.0</td>
      <td>205.0</td>
      <td>0.0</td>
      <td>4362.0</td>
      <td>185.0</td>
      <td>4404.0</td>
      <td>36798.0</td>
      <td>NaN</td>
      <td>66.19</td>
      <td>70.13</td>
    </tr>
    <tr>
      <th>2015-10-02 09:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2513.0</td>
      <td>3690.0</td>
      <td>38921.0</td>
      <td>NaN</td>
      <td>70.09</td>
      <td>70.49</td>
    </tr>
    <tr>
      <th>2015-12-02 08:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>684.0</td>
      <td>1510.0</td>
      <td>37413.0</td>
      <td>NaN</td>
      <td>75.71</td>
      <td>80.44</td>
    </tr>
    <tr>
      <th>2016-04-13 03:00:00+00:00</th>
      <td>220.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3390.0</td>
      <td>1242.0</td>
      <td>243.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2270.0</td>
      <td>...</td>
      <td>150.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>8596.0</td>
      <td>70.0</td>
      <td>8557.0</td>
      <td>23514.0</td>
      <td>23614.0</td>
      <td>18.69</td>
      <td>25.14</td>
    </tr>
    <tr>
      <th>2016-04-25 03:00:00+00:00</th>
      <td>190.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2969.0</td>
      <td>886.0</td>
      <td>151.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1340.0</td>
      <td>...</td>
      <td>454.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>5989.0</td>
      <td>412.0</td>
      <td>6019.0</td>
      <td>21471.0</td>
      <td>NaN</td>
      <td>15.00</td>
      <td>22.65</td>
    </tr>
    <tr>
      <th>2016-04-25 05:00:00+00:00</th>
      <td>206.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3673.0</td>
      <td>1143.0</td>
      <td>185.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>...</td>
      <td>283.0</td>
      <td>214.0</td>
      <td>0.0</td>
      <td>5682.0</td>
      <td>208.0</td>
      <td>5758.0</td>
      <td>27635.0</td>
      <td>NaN</td>
      <td>32.97</td>
      <td>40.18</td>
    </tr>
    <tr>
      <th>2016-05-10 21:00:00+00:00</th>
      <td>348.0</td>
      <td>960.0</td>
      <td>0.0</td>
      <td>6800.0</td>
      <td>5219.0</td>
      <td>299.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>58.0</td>
      <td>280.0</td>
      <td>0.0</td>
      <td>3311.0</td>
      <td>84.0</td>
      <td>3465.0</td>
      <td>26641.0</td>
      <td>NaN</td>
      <td>51.57</td>
      <td>39.11</td>
    </tr>
    <tr>
      <th>2016-06-11 23:00:00+00:00</th>
      <td>356.0</td>
      <td>595.0</td>
      <td>0.0</td>
      <td>5719.0</td>
      <td>6165.0</td>
      <td>274.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>382.0</td>
      <td>...</td>
      <td>30.0</td>
      <td>291.0</td>
      <td>0.0</td>
      <td>2019.0</td>
      <td>9.0</td>
      <td>2064.0</td>
      <td>24715.0</td>
      <td>24155.0</td>
      <td>60.23</td>
      <td>48.72</td>
    </tr>
    <tr>
      <th>2016-07-09 20:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>499.0</td>
      <td>4914.0</td>
      <td>34985.0</td>
      <td>NaN</td>
      <td>45.72</td>
      <td>51.72</td>
    </tr>
    <tr>
      <th>2016-07-11 22:00:00+00:00</th>
      <td>346.0</td>
      <td>595.0</td>
      <td>0.0</td>
      <td>5951.0</td>
      <td>6131.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>494.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>309.0</td>
      <td>0.0</td>
      <td>2031.0</td>
      <td>10.0</td>
      <td>2050.0</td>
      <td>25313.0</td>
      <td>25103.0</td>
      <td>64.99</td>
      <td>47.49</td>
    </tr>
    <tr>
      <th>2016-09-28 07:00:00+00:00</th>
      <td>347.0</td>
      <td>594.0</td>
      <td>0.0</td>
      <td>5522.0</td>
      <td>6272.0</td>
      <td>292.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>982.0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>5478.0</td>
      <td>967.0</td>
      <td>5359.0</td>
      <td>31072.0</td>
      <td>NaN</td>
      <td>49.72</td>
      <td>56.40</td>
    </tr>
    <tr>
      <th>2016-10-11 21:00:00+00:00</th>
      <td>342.0</td>
      <td>894.0</td>
      <td>0.0</td>
      <td>8583.0</td>
      <td>5899.0</td>
      <td>299.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>...</td>
      <td>51.0</td>
      <td>299.0</td>
      <td>0.0</td>
      <td>4720.0</td>
      <td>13.0</td>
      <td>4737.0</td>
      <td>28348.0</td>
      <td>28465.0</td>
      <td>58.72</td>
      <td>60.98</td>
    </tr>
    <tr>
      <th>2016-10-27 21:00:00+00:00</th>
      <td>351.0</td>
      <td>554.0</td>
      <td>0.0</td>
      <td>7176.0</td>
      <td>5690.0</td>
      <td>321.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>70.0</td>
      <td>299.0</td>
      <td>0.0</td>
      <td>3193.0</td>
      <td>46.0</td>
      <td>3143.0</td>
      <td>26423.0</td>
      <td>26583.0</td>
      <td>55.70</td>
      <td>62.84</td>
    </tr>
    <tr>
      <th>2016-11-23 03:00:00+00:00</th>
      <td>NaN</td>
      <td>900.0</td>
      <td>0.0</td>
      <td>4838.0</td>
      <td>4547.0</td>
      <td>269.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1413.0</td>
      <td>...</td>
      <td>15.0</td>
      <td>227.0</td>
      <td>0.0</td>
      <td>4598.0</td>
      <td>3.0</td>
      <td>4566.0</td>
      <td>23469.0</td>
      <td>23112.0</td>
      <td>43.19</td>
      <td>49.11</td>
    </tr>
    <tr>
      <th>2017-11-14 11:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10064.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4325.0</td>
      <td>7561.0</td>
      <td>33805.0</td>
      <td>NaN</td>
      <td>60.53</td>
      <td>66.17</td>
    </tr>
    <tr>
      <th>2017-11-14 18:00:00+00:00</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12336.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>128.0</td>
      <td>5679.0</td>
      <td>35592.0</td>
      <td>NaN</td>
      <td>68.05</td>
      <td>75.45</td>
    </tr>
    <tr>
      <th>2018-06-11 16:00:00+00:00</th>
      <td>331.0</td>
      <td>506.0</td>
      <td>0.0</td>
      <td>7538.0</td>
      <td>5360.0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>170.0</td>
      <td>269.0</td>
      <td>0.0</td>
      <td>9165.0</td>
      <td>125.0</td>
      <td>10329.0</td>
      <td>34752.0</td>
      <td>NaN</td>
      <td>69.87</td>
      <td>64.93</td>
    </tr>
    <tr>
      <th>2018-07-11 07:00:00+00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>849.0</td>
      <td>9956.0</td>
      <td>33938.0</td>
      <td>NaN</td>
      <td>63.01</td>
      <td>69.79</td>
    </tr>
  </tbody>
</table>
<p>47 rows × 26 columns</p>
</div>




```python
# Forward filling missing values
df_e3 = df_e2.fillna(method='ffill')
```


```python
# Nice! no more missing values
df_e3.isna().sum()
```




    generation biomass                             0
    generation fossil brown coal/lignite           0
    generation fossil coal-derived gas             0
    generation fossil gas                          0
    generation fossil hard coal                    0
    generation fossil oil                          0
    generation fossil oil shale                    0
    generation fossil peat                         0
    generation geothermal                          0
    generation hydro pumped storage consumption    0
    generation hydro run-of-river and poundage     0
    generation hydro water reservoir               0
    generation marine                              0
    generation nuclear                             0
    generation other                               0
    generation other renewable                     0
    generation solar                               0
    generation waste                               0
    generation wind offshore                       0
    generation wind onshore                        0
    forecast solar day ahead                       0
    forecast wind onshore day ahead                0
    total load forecast                            0
    total load actual                              0
    price day ahead                                0
    price actual                                   0
    dtype: int64




```python
pd.set_option('display.max_columns', 11)
```


```python
df_e3.head(8)
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
      <th>generation biomass</th>
      <th>generation fossil brown coal/lignite</th>
      <th>generation fossil coal-derived gas</th>
      <th>generation fossil gas</th>
      <th>generation fossil hard coal</th>
      <th>...</th>
      <th>forecast wind onshore day ahead</th>
      <th>total load forecast</th>
      <th>total load actual</th>
      <th>price day ahead</th>
      <th>price actual</th>
    </tr>
    <tr>
      <th>time</th>
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
      <th>2014-12-31 23:00:00+00:00</th>
      <td>447.0</td>
      <td>329.0</td>
      <td>0.0</td>
      <td>4844.0</td>
      <td>4821.0</td>
      <td>...</td>
      <td>6436.0</td>
      <td>26118.0</td>
      <td>25385.0</td>
      <td>50.10</td>
      <td>65.41</td>
    </tr>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>449.0</td>
      <td>328.0</td>
      <td>0.0</td>
      <td>5196.0</td>
      <td>4755.0</td>
      <td>...</td>
      <td>5856.0</td>
      <td>24934.0</td>
      <td>24382.0</td>
      <td>48.10</td>
      <td>64.92</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>448.0</td>
      <td>323.0</td>
      <td>0.0</td>
      <td>4857.0</td>
      <td>4581.0</td>
      <td>...</td>
      <td>5454.0</td>
      <td>23515.0</td>
      <td>22734.0</td>
      <td>47.33</td>
      <td>64.48</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>438.0</td>
      <td>254.0</td>
      <td>0.0</td>
      <td>4314.0</td>
      <td>4131.0</td>
      <td>...</td>
      <td>5151.0</td>
      <td>22642.0</td>
      <td>21286.0</td>
      <td>42.27</td>
      <td>59.32</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>428.0</td>
      <td>187.0</td>
      <td>0.0</td>
      <td>4130.0</td>
      <td>3840.0</td>
      <td>...</td>
      <td>4861.0</td>
      <td>21785.0</td>
      <td>20264.0</td>
      <td>38.41</td>
      <td>56.04</td>
    </tr>
    <tr>
      <th>2015-01-01 04:00:00+00:00</th>
      <td>410.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>4038.0</td>
      <td>3590.0</td>
      <td>...</td>
      <td>4617.0</td>
      <td>21441.0</td>
      <td>19905.0</td>
      <td>35.72</td>
      <td>53.63</td>
    </tr>
    <tr>
      <th>2015-01-01 05:00:00+00:00</th>
      <td>401.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>4040.0</td>
      <td>3368.0</td>
      <td>...</td>
      <td>4276.0</td>
      <td>21285.0</td>
      <td>20010.0</td>
      <td>35.13</td>
      <td>51.73</td>
    </tr>
    <tr>
      <th>2015-01-01 06:00:00+00:00</th>
      <td>408.0</td>
      <td>172.0</td>
      <td>0.0</td>
      <td>4030.0</td>
      <td>3208.0</td>
      <td>...</td>
      <td>3994.0</td>
      <td>21545.0</td>
      <td>20377.0</td>
      <td>36.22</td>
      <td>51.43</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>



## Decomposition (Daily)


```python
df_e_daily = df_e2.resample("D").mean()

df_e_daily.head()
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
      <th>generation biomass</th>
      <th>generation fossil brown coal/lignite</th>
      <th>generation fossil coal-derived gas</th>
      <th>generation fossil gas</th>
      <th>generation fossil hard coal</th>
      <th>...</th>
      <th>forecast wind onshore day ahead</th>
      <th>total load forecast</th>
      <th>total load actual</th>
      <th>price day ahead</th>
      <th>price actual</th>
    </tr>
    <tr>
      <th>time</th>
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
      <th>2014-12-31 00:00:00+00:00</th>
      <td>447.000000</td>
      <td>329.000000</td>
      <td>0.0</td>
      <td>4844.000000</td>
      <td>4821.000000</td>
      <td>...</td>
      <td>6436.000000</td>
      <td>26118.000000</td>
      <td>25385.000000</td>
      <td>50.100000</td>
      <td>65.410000</td>
    </tr>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>425.208333</td>
      <td>243.708333</td>
      <td>0.0</td>
      <td>4187.791667</td>
      <td>4099.458333</td>
      <td>...</td>
      <td>3942.416667</td>
      <td>24753.250000</td>
      <td>23966.958333</td>
      <td>45.031667</td>
      <td>62.090833</td>
    </tr>
    <tr>
      <th>2015-01-02 00:00:00+00:00</th>
      <td>389.875000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3566.166667</td>
      <td>1258.125000</td>
      <td>...</td>
      <td>11117.041667</td>
      <td>27519.416667</td>
      <td>27188.541667</td>
      <td>17.598333</td>
      <td>69.443750</td>
    </tr>
    <tr>
      <th>2015-01-03 00:00:00+00:00</th>
      <td>436.875000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3451.791667</td>
      <td>1375.958333</td>
      <td>...</td>
      <td>9113.416667</td>
      <td>25500.833333</td>
      <td>25097.750000</td>
      <td>25.342083</td>
      <td>65.223333</td>
    </tr>
    <tr>
      <th>2015-01-04 00:00:00+00:00</th>
      <td>396.375000</td>
      <td>13.583333</td>
      <td>0.0</td>
      <td>3526.125000</td>
      <td>2315.291667</td>
      <td>...</td>
      <td>8022.833333</td>
      <td>27167.875000</td>
      <td>27104.916667</td>
      <td>30.658333</td>
      <td>58.912083</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Hard to pick out much of a trend in daily load measurements and a monthlong timescale
df_e_daily['total load actual'][30:60].plot(figsize=(20, 10))
plt.title('Daily Average Electricity Demand')
plt.show()
```


    
![png](output_31_0.png)
    



```python
# The trend seen here has electicity usage peaking near in the new year (cold, heating) and about halfway through (hot, AC)
df_e_daily['total load actual'][0:365].plot(figsize=(20, 10))
plt.title('Daily Average Electricity Demand')
plt.show()
```


    
![png](output_32_0.png)
    



```python
# Taking monthly mean to find trends throughout the year
month_mean = df_e_daily.groupby(df_e_daily.index.month_name()).mean()

# Reordering the Months Correctly
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_mean = month_mean.reindex(new_order, axis=0)
```


```python
plt.figure()

plt.scatter(month_mean.index, month_mean['total load actual'])

plt.xticks(rotation=30)

plt.show()
```


    
![png](output_34_0.png)
    


Decomposition: One Year Time Horizon


```python
# Limit decomposition to just one year of data
decomposition2 = tsa.seasonal_decompose(df_e_daily['total load actual'][0:365], model='additive')
```


```python
decomposition2.plot()
plt.show()
```


    
![png](output_37_0.png)
    



```python

```


```python

```


```python

```

## Decomposition: One Month Time Horizon


```python
# Limit decomposition to just one year of data
decomposition2 = tsa.seasonal_decompose(df_e3['total load actual'], model='additive')
```


```python
#df_e3['total load actual'][13500:14220].plot(figsize=(20, 10))
#plt.title('Electricity Demand')
#plt.show()
```


```python
#df_e3["Trend"] = decomposition2.trend
#df_e3["Seasonal"] = decomposition2.seasonal
#df_e3["Residual"] = decomposition2.resid
```


```python
decomposition2.plot()
plt.show()
```


    
![png](output_45_0.png)
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## FB Prophet TimeSeries Model

### Cleaning Dataframe for Prophet


```python
# This new dataframe is simplified for timeseries forecasting. Only contains a datetime and a load value.
df_prophet = pd.DataFrame()
df_prophet['y'] = df_e3['total load actual']
df_prophet['ds'] = df_e3.index
```


```python
df_prophet
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
      <th>y</th>
      <th>ds</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-31 23:00:00+00:00</th>
      <td>25385.0</td>
      <td>2014-12-31 23:00:00+00:00</td>
    </tr>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>24382.0</td>
      <td>2015-01-01 00:00:00+00:00</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>22734.0</td>
      <td>2015-01-01 01:00:00+00:00</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>21286.0</td>
      <td>2015-01-01 02:00:00+00:00</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>20264.0</td>
      <td>2015-01-01 03:00:00+00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-31 18:00:00+00:00</th>
      <td>30653.0</td>
      <td>2018-12-31 18:00:00+00:00</td>
    </tr>
    <tr>
      <th>2018-12-31 19:00:00+00:00</th>
      <td>29735.0</td>
      <td>2018-12-31 19:00:00+00:00</td>
    </tr>
    <tr>
      <th>2018-12-31 20:00:00+00:00</th>
      <td>28071.0</td>
      <td>2018-12-31 20:00:00+00:00</td>
    </tr>
    <tr>
      <th>2018-12-31 21:00:00+00:00</th>
      <td>25801.0</td>
      <td>2018-12-31 21:00:00+00:00</td>
    </tr>
    <tr>
      <th>2018-12-31 22:00:00+00:00</th>
      <td>24455.0</td>
      <td>2018-12-31 22:00:00+00:00</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 2 columns</p>
</div>




```python
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
```


```python
df_prophet.reset_index(inplace=True)
```


```python
df_prophet = df_prophet.drop(['time'], axis=1)
```


```python
df_prophet
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
      <th>y</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25385.0</td>
      <td>2014-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24382.0</td>
      <td>2015-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22734.0</td>
      <td>2015-01-01 01:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21286.0</td>
      <td>2015-01-01 02:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20264.0</td>
      <td>2015-01-01 03:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35059</th>
      <td>30653.0</td>
      <td>2018-12-31 18:00:00</td>
    </tr>
    <tr>
      <th>35060</th>
      <td>29735.0</td>
      <td>2018-12-31 19:00:00</td>
    </tr>
    <tr>
      <th>35061</th>
      <td>28071.0</td>
      <td>2018-12-31 20:00:00</td>
    </tr>
    <tr>
      <th>35062</th>
      <td>25801.0</td>
      <td>2018-12-31 21:00:00</td>
    </tr>
    <tr>
      <th>35063</th>
      <td>24455.0</td>
      <td>2018-12-31 22:00:00</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 2 columns</p>
</div>



Dataframe 'df_prophet' is now ready for the prophet model

### Modeling with 'Walk Forward' Validation


```python
# Need to use a rolling dataframe? Trying it here, may need to implement 'pipeline' now
# Dataframe formatted for prophet
df_prophet

# Number of records in training set. This is the first three years of entries, and will be included in every train set
n_train = 26306 # 3 Years, Test set starts 1/1/2018 at midnight (first datapoint at 1AM), which also happens to be a monday
n_records = len(df_prophet)
```


```python
# The test window will step in one week intervals - 168 hours
week_steps = range(n_train, n_records-168, 168)

# MAPE score for each week of the test year will be stored here
Scores = []

# Counter used to track progress of the loop
week = 1

# For loop to test all 52 weeks of the final year of data
for n in week_steps:
    # Train data encompasses the start of the dataset to the start of sliding test set
    train = df_prophet[0:n]
    # Test data is 168 hours- or one week long
    test = df_prophet[n:n+168]
    
    # Instantiate Model
    model = Prophet()
    # Fit Model
    model = model.fit(train)
    
    #Evaluate Model
    predict = model.predict(pd.DataFrame({'ds':test['ds']}))
    y_actual = test['y']
    y_predicted = predict['yhat'].astype(int)
    Prophet_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    
    print(f' Week {week} MAPE Score: {Prophet_MAPE}')
    Scores.append(Prophet_MAPE)
    week = week+1

print(f' Combined MAPE Score: {sum(Scores)/len(Scores)}')
```

    21:17:05 - cmdstanpy - INFO - Chain [1] start processing
    21:17:10 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 1 MAPE Score: 0.15845077917533115
    

    21:17:12 - cmdstanpy - INFO - Chain [1] start processing
    21:17:16 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 2 MAPE Score: 0.09208173148719498
    

    21:17:18 - cmdstanpy - INFO - Chain [1] start processing
    21:17:25 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 3 MAPE Score: 0.06150220123754084
    

    21:17:27 - cmdstanpy - INFO - Chain [1] start processing
    21:17:32 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 4 MAPE Score: 0.04744363873019493
    

    21:17:34 - cmdstanpy - INFO - Chain [1] start processing
    21:17:43 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 5 MAPE Score: 0.07199989882346654
    

    21:17:45 - cmdstanpy - INFO - Chain [1] start processing
    21:17:51 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 6 MAPE Score: 0.08960798708044246
    

    21:17:53 - cmdstanpy - INFO - Chain [1] start processing
    21:18:00 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 7 MAPE Score: 0.08623689964950326
    

    21:18:02 - cmdstanpy - INFO - Chain [1] start processing
    21:18:07 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 8 MAPE Score: 0.05020438767462866
    

    21:18:09 - cmdstanpy - INFO - Chain [1] start processing
    21:18:14 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 9 MAPE Score: 0.054774911910416
    

    21:18:15 - cmdstanpy - INFO - Chain [1] start processing
    21:18:21 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 10 MAPE Score: 0.09799577771065464
    

    21:18:23 - cmdstanpy - INFO - Chain [1] start processing
    21:18:32 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 11 MAPE Score: 0.05448513969727742
    

    21:18:34 - cmdstanpy - INFO - Chain [1] start processing
    21:18:41 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 12 MAPE Score: 0.06935319641724494
    

    21:18:43 - cmdstanpy - INFO - Chain [1] start processing
    21:18:52 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 13 MAPE Score: 0.12439538522388853
    

    21:18:54 - cmdstanpy - INFO - Chain [1] start processing
    21:19:00 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 14 MAPE Score: 0.08956423045760484
    

    21:19:02 - cmdstanpy - INFO - Chain [1] start processing
    21:19:09 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 15 MAPE Score: 0.07645889428500117
    

    21:19:10 - cmdstanpy - INFO - Chain [1] start processing
    21:19:16 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 16 MAPE Score: 0.05016634806454564
    

    21:19:18 - cmdstanpy - INFO - Chain [1] start processing
    21:19:23 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 17 MAPE Score: 0.05529232224054959
    

    21:19:25 - cmdstanpy - INFO - Chain [1] start processing
    21:19:31 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 18 MAPE Score: 0.07867404676725143
    

    21:19:33 - cmdstanpy - INFO - Chain [1] start processing
    21:19:39 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 19 MAPE Score: 0.07819794788917286
    

    21:19:41 - cmdstanpy - INFO - Chain [1] start processing
    21:19:52 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 20 MAPE Score: 0.055315003044599347
    

    21:19:54 - cmdstanpy - INFO - Chain [1] start processing
    21:20:00 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 21 MAPE Score: 0.04538455153303415
    

    21:20:02 - cmdstanpy - INFO - Chain [1] start processing
    21:20:08 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 22 MAPE Score: 0.09709770520271517
    

    21:20:10 - cmdstanpy - INFO - Chain [1] start processing
    21:20:17 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 23 MAPE Score: 0.09263505137305209
    

    21:20:19 - cmdstanpy - INFO - Chain [1] start processing
    21:20:27 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 24 MAPE Score: 0.0972162438423737
    

    21:20:29 - cmdstanpy - INFO - Chain [1] start processing
    21:20:36 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 25 MAPE Score: 0.04061805653748827
    

    21:20:38 - cmdstanpy - INFO - Chain [1] start processing
    21:20:47 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 26 MAPE Score: 0.05065821448360212
    

    21:20:49 - cmdstanpy - INFO - Chain [1] start processing
    21:21:00 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 27 MAPE Score: 0.09870948907709762
    

    21:21:02 - cmdstanpy - INFO - Chain [1] start processing
    21:21:08 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 28 MAPE Score: 0.11228554594062846
    

    21:21:10 - cmdstanpy - INFO - Chain [1] start processing
    21:21:22 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 29 MAPE Score: 0.061175173951927
    

    21:21:24 - cmdstanpy - INFO - Chain [1] start processing
    21:21:34 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 30 MAPE Score: 0.042930681255789534
    

    21:21:36 - cmdstanpy - INFO - Chain [1] start processing
    21:21:44 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 31 MAPE Score: 0.08872436755521362
    

    21:21:46 - cmdstanpy - INFO - Chain [1] start processing
    21:21:55 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 32 MAPE Score: 0.08330075462567198
    

    21:21:57 - cmdstanpy - INFO - Chain [1] start processing
    21:22:06 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 33 MAPE Score: 0.055111824762106576
    

    21:22:08 - cmdstanpy - INFO - Chain [1] start processing
    21:22:16 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 34 MAPE Score: 0.040622046634869985
    

    21:22:18 - cmdstanpy - INFO - Chain [1] start processing
    21:22:26 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 35 MAPE Score: 0.08663002970234845
    

    21:22:28 - cmdstanpy - INFO - Chain [1] start processing
    21:22:36 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 36 MAPE Score: 0.09706255850107731
    

    21:22:38 - cmdstanpy - INFO - Chain [1] start processing
    21:22:49 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 37 MAPE Score: 0.06714868869582512
    

    21:22:51 - cmdstanpy - INFO - Chain [1] start processing
    21:22:59 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 38 MAPE Score: 0.06014645956795657
    

    21:23:01 - cmdstanpy - INFO - Chain [1] start processing
    21:23:10 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 39 MAPE Score: 0.05745001591411277
    

    21:23:12 - cmdstanpy - INFO - Chain [1] start processing
    21:23:18 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 40 MAPE Score: 0.10551743018375494
    

    21:23:20 - cmdstanpy - INFO - Chain [1] start processing
    21:23:27 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 41 MAPE Score: 0.07983595191865768
    

    21:23:29 - cmdstanpy - INFO - Chain [1] start processing
    21:23:39 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 42 MAPE Score: 0.044837524525185135
    

    21:23:41 - cmdstanpy - INFO - Chain [1] start processing
    21:23:49 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 43 MAPE Score: 0.03958187294746563
    

    21:23:51 - cmdstanpy - INFO - Chain [1] start processing
    21:23:59 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 44 MAPE Score: 0.08969724289806821
    

    21:24:01 - cmdstanpy - INFO - Chain [1] start processing
    21:24:10 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 45 MAPE Score: 0.06758954509279157
    

    21:24:12 - cmdstanpy - INFO - Chain [1] start processing
    21:24:20 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 46 MAPE Score: 0.04924304857678757
    

    21:24:22 - cmdstanpy - INFO - Chain [1] start processing
    21:24:30 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 47 MAPE Score: 0.045552090343131006
    

    21:24:33 - cmdstanpy - INFO - Chain [1] start processing
    21:24:38 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 48 MAPE Score: 0.07319704676655757
    

    21:24:40 - cmdstanpy - INFO - Chain [1] start processing
    21:24:47 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 49 MAPE Score: 0.09492409391240281
    

    21:24:49 - cmdstanpy - INFO - Chain [1] start processing
    21:24:57 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 50 MAPE Score: 0.0796675225937794
    

    21:25:00 - cmdstanpy - INFO - Chain [1] start processing
    21:25:06 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 51 MAPE Score: 0.059947832475541564
    

    21:25:09 - cmdstanpy - INFO - Chain [1] start processing
    21:25:26 - cmdstanpy - INFO - Chain [1] done processing
    

     Week 52 MAPE Score: 0.10707969494845848
     Combined MAPE Score: 0.0741111362289612
    


```python

```


```python

```


```python

```


```python

```

### Single run of the model


```python
# Quick test of EDA
test
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
      <th>y</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34874</th>
      <td>20759.0</td>
      <td>2018-12-24 01:00:00</td>
    </tr>
    <tr>
      <th>34875</th>
      <td>20087.0</td>
      <td>2018-12-24 02:00:00</td>
    </tr>
    <tr>
      <th>34876</th>
      <td>19894.0</td>
      <td>2018-12-24 03:00:00</td>
    </tr>
    <tr>
      <th>34877</th>
      <td>20299.0</td>
      <td>2018-12-24 04:00:00</td>
    </tr>
    <tr>
      <th>34878</th>
      <td>21617.0</td>
      <td>2018-12-24 05:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35037</th>
      <td>30229.0</td>
      <td>2018-12-30 20:00:00</td>
    </tr>
    <tr>
      <th>35038</th>
      <td>29145.0</td>
      <td>2018-12-30 21:00:00</td>
    </tr>
    <tr>
      <th>35039</th>
      <td>26934.0</td>
      <td>2018-12-30 22:00:00</td>
    </tr>
    <tr>
      <th>35040</th>
      <td>24312.0</td>
      <td>2018-12-30 23:00:00</td>
    </tr>
    <tr>
      <th>35041</th>
      <td>22140.0</td>
      <td>2018-12-31 00:00:00</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 2 columns</p>
</div>




```python
# Instantiate Model
model = Prophet()
# Fit Model
model = model.fit(train)
```

    21:25:28 - cmdstanpy - INFO - Chain [1] start processing
    21:25:45 - cmdstanpy - INFO - Chain [1] done processing
    


```python
predict = model.predict(pd.DataFrame({'ds':test['ds']}))
y_actual = test['y']
y_predicted = predict['yhat'].astype(int)
Prophet_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
```


```python
y_predicted
```




    0      22128
    1      21865
    2      22207
    3      23347
    4      25248
           ...  
    163    28848
    164    27163
    165    25167
    166    23435
    167    22222
    Name: yhat, Length: 168, dtype: int32




```python
Prophet_MAPE
```




    0.10707969494845848




```python

```


```python

```


```python

```

### Plotting


```python
y_actual_reset = y_actual.reset_index()
```


```python
y_remainder = y_predicted.subtract(y_actual_reset['y'])
y_remainder
```




    0      1369.0
    1      1778.0
    2      2313.0
    3      3048.0
    4      3631.0
            ...  
    163   -1381.0
    164   -1982.0
    165   -1767.0
    166    -877.0
    167      82.0
    Length: 168, dtype: float64




```python
df_prophet
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
      <th>y</th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25385.0</td>
      <td>2014-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24382.0</td>
      <td>2015-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22734.0</td>
      <td>2015-01-01 01:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21286.0</td>
      <td>2015-01-01 02:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20264.0</td>
      <td>2015-01-01 03:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35059</th>
      <td>30653.0</td>
      <td>2018-12-31 18:00:00</td>
    </tr>
    <tr>
      <th>35060</th>
      <td>29735.0</td>
      <td>2018-12-31 19:00:00</td>
    </tr>
    <tr>
      <th>35061</th>
      <td>28071.0</td>
      <td>2018-12-31 20:00:00</td>
    </tr>
    <tr>
      <th>35062</th>
      <td>25801.0</td>
      <td>2018-12-31 21:00:00</td>
    </tr>
    <tr>
      <th>35063</th>
      <td>24455.0</td>
      <td>2018-12-31 22:00:00</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 2 columns</p>
</div>




```python
plt.figure(figsize=(11,6))


# Left and Right ends of the plot line up with the test set
left = df_prophet['ds'][n:n+1]
right = df_prophet['ds'][n+168:n+169]


# These are the forecasted values for the next 100 hours
plt.plot(test['ds'], y_actual, color='gold', label='Total Load Actual')

plt.plot(test['ds'], y_predicted, color='cornflowerblue', label='Total Load Predicted')

plt.plot(test['ds'], y_remainder, color='green', label='Remainder')

plt.xlim([left, right])

plt.axhline(color='yellow', label='Zero Line')

plt.axhline(min(y_remainder), color='black', label= f'2 Week Minimum: {min(y_remainder)}')

plt.legend()

plt.ylabel('Total Electrical Load (MW)')

plt.show()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~\anaconda3\envs\timeseries\lib\site-packages\pandas\core\indexes\base.py:3629, in Index.get_loc(self, key, method, tolerance)
       3628 try:
    -> 3629     return self._engine.get_loc(casted_key)
       3630 except KeyError as err:
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\pandas\_libs\index.pyx:136, in pandas._libs.index.IndexEngine.get_loc()
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\pandas\_libs\index.pyx:163, in pandas._libs.index.IndexEngine.get_loc()
    

    File pandas\_libs\hashtable_class_helper.pxi:5198, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    File pandas\_libs\hashtable_class_helper.pxi:5206, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'ds'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    Input In [98], in <cell line: 10>()
          6 right = df_prophet['ds'][n+168:n+169]
          9 # These are the forecasted values for the next 100 hours
    ---> 10 plt.plot(test['ds'], y_actual, color='gold', label='Total Load Actual')
         12 plt.plot(test['ds'], y_predicted, color='cornflowerblue', label='Total Load Predicted')
         14 plt.plot(test['ds'], y_remainder, color='green', label='Remainder')
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\pandas\core\frame.py:3505, in DataFrame.__getitem__(self, key)
       3503 if self.columns.nlevels > 1:
       3504     return self._getitem_multilevel(key)
    -> 3505 indexer = self.columns.get_loc(key)
       3506 if is_integer(indexer):
       3507     indexer = [indexer]
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\pandas\core\indexes\base.py:3631, in Index.get_loc(self, key, method, tolerance)
       3629     return self._engine.get_loc(casted_key)
       3630 except KeyError as err:
    -> 3631     raise KeyError(key) from err
       3632 except TypeError:
       3633     # If we have a listlike key, _check_indexing_error will raise
       3634     #  InvalidIndexError. Otherwise we fall through and re-raise
       3635     #  the TypeError.
       3636     self._check_indexing_error(key)
    

    KeyError: 'ds'



    <Figure size 1100x600 with 0 Axes>



```python

```


```python

```


```python

```


```python

```

## (Seasonal?) Autoregressive Integrated Moving Average Model (ARIMA)

### Data Cleanup


```python
df_AR = pd.DataFrame(df_e3['total load actual'])
df_AR
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
      <th>total load actual</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-31 23:00:00+00:00</th>
      <td>25385.0</td>
    </tr>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>24382.0</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>22734.0</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>21286.0</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>20264.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-31 18:00:00+00:00</th>
      <td>30653.0</td>
    </tr>
    <tr>
      <th>2018-12-31 19:00:00+00:00</th>
      <td>29735.0</td>
    </tr>
    <tr>
      <th>2018-12-31 20:00:00+00:00</th>
      <td>28071.0</td>
    </tr>
    <tr>
      <th>2018-12-31 21:00:00+00:00</th>
      <td>25801.0</td>
    </tr>
    <tr>
      <th>2018-12-31 22:00:00+00:00</th>
      <td>24455.0</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 1 columns</p>
</div>



### Single Run Modeling


```python
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
```


```python
# Looks like there is correlation through the first 5-7 lagged measurments
autocorrelation_plot(df_AR[503:528])
```




    <AxesSubplot:xlabel='Lag', ylabel='Autocorrelation'>




    
![png](output_97_1.png)
    



```python
model_ARIMA = ARIMA(df_AR, order=(6,1,0))
ARIMA_fit = model_ARIMA.fit()
```


```python
ARIMA_fit.forecast(168)
```




    2018-12-31 23:00:00+00:00    23371.238334
    2019-01-01 00:00:00+00:00    22749.499952
    2019-01-01 01:00:00+00:00    22715.003686
    2019-01-01 02:00:00+00:00    23167.131562
    2019-01-01 03:00:00+00:00    23894.586653
                                     ...     
    2019-01-07 18:00:00+00:00    24970.707621
    2019-01-07 19:00:00+00:00    24970.707620
    2019-01-07 20:00:00+00:00    24970.707620
    2019-01-07 21:00:00+00:00    24970.707620
    2019-01-07 22:00:00+00:00    24970.707620
    Freq: H, Name: predicted_mean, Length: 168, dtype: float64




```python
print(ARIMA_fit.summary())
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:      total load actual   No. Observations:                35064
    Model:                 ARIMA(6, 1, 0)   Log Likelihood             -290529.615
    Date:                Fri, 09 Dec 2022   AIC                         581073.229
    Time:                        21:25:49   BIC                         581132.483
    Sample:                    12-31-2014   HQIC                        581092.101
                             - 12-31-2018                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.7637      0.002    330.752      0.000       0.759       0.768
    ar.L2         -0.1176      0.003    -40.889      0.000      -0.123      -0.112
    ar.L3         -0.0135      0.006     -2.313      0.021      -0.025      -0.002
    ar.L4         -0.0302      0.008     -3.958      0.000      -0.045      -0.015
    ar.L5         -0.0803      0.007    -11.351      0.000      -0.094      -0.066
    ar.L6         -0.1008      0.005    -19.235      0.000      -0.111      -0.091
    sigma2      9.182e+05   2647.897    346.770      0.000    9.13e+05    9.23e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   1.66   Jarque-Bera (JB):           1837942.08
    Prob(Q):                              0.20   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.81   Skew:                             0.40
    Prob(H) (two-sided):                  0.00   Kurtosis:                        38.46
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
residuals = pd.DataFrame(ARIMA_fit.resid)[25:50]
residuals.plot()
plt.show()
```


    
![png](output_101_0.png)
    



```python

```

### Rolling Train/Test Split


```python
# Using a rolling dataframe!
# Dataframe formatted for AR model
df_AR

# Number of records in training set. This is the first three years of entries, and will be included in every train set
n_train = 26306 # 3 Years, Test set starts 1/1/2018 at midnight (first datapoint at 1AM), which also happens to be a monday
n_records = len(df_AR)
```


```python
# The test window will step in one week intervals - 168 hours
week_steps = range(n_train, n_records-168, 168)

# MAPE score for each week of the test year will be stored here
Scores = []

# Counter used to track progress of the loop
week = 1

# For loop to test all 52 weeks of the final year of data
for n in week_steps:
    # Train data encompasses the start of the dataset to the start of sliding test set
    train = df_AR[0:n]
    # Test data is 168 hours- or one week long
    test = df_AR[n:n+168]
    
    # Instantiate Model
    model_ARIMA = ARIMA(train, order=(6,1,0))
    # Fit Model
    ARIMA_fit = model_ARIMA.fit()
    
    #Evaluate Model
    predict = ARIMA_fit.forecast(168)
    y_actual = test['total load actual']
    y_predicted = predict.astype(float)
    Prophet_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    
    print(f' Week {week} MAPE Score: {Prophet_MAPE}')
    Scores.append(Prophet_MAPE)
    week = week+1

print(f' Combined MAPE Score: {sum(Scores)/len(Scores)}')
```

     Week 1 MAPE Score: 0.15185196744487964
     Week 2 MAPE Score: 0.15539206873108466
     Week 3 MAPE Score: 0.17023944214032757
     Week 4 MAPE Score: 0.18024514739404887
     Week 5 MAPE Score: 0.18906565760881988
     Week 6 MAPE Score: 0.17627782219802923
     Week 7 MAPE Score: 0.18022045726359362
     Week 8 MAPE Score: 0.16848172675365775
     Week 9 MAPE Score: 0.17603374364092203
     Week 10 MAPE Score: 0.15141423211592353
     Week 11 MAPE Score: 0.1855753049020891
     Week 12 MAPE Score: 0.18543998019877755
     Week 13 MAPE Score: 0.12992305068186008
     Week 14 MAPE Score: 0.13626366528706835
     Week 15 MAPE Score: 0.13397899201262914
     Week 16 MAPE Score: 0.15152010348776565
     Week 17 MAPE Score: 0.1687394203865806
     Week 18 MAPE Score: 0.22794596661613425
     Week 19 MAPE Score: 0.15215048263490932
     Week 20 MAPE Score: 0.15991704971761744
     Week 21 MAPE Score: 0.1704742705297432
     Week 22 MAPE Score: 0.22318693920118018
     Week 23 MAPE Score: 0.13505306323248092
     Week 24 MAPE Score: 0.13660188743966953
     Week 25 MAPE Score: 0.20276487746810773
     Week 26 MAPE Score: 0.17693720767160306
     Week 27 MAPE Score: 0.1356215081294701
     Week 28 MAPE Score: 0.12655765014414536
     Week 29 MAPE Score: 0.17879016893631294
     Week 30 MAPE Score: 0.1988911979585692
     Week 31 MAPE Score: 0.19680713254377286
     Week 32 MAPE Score: 0.13161454129404984
     Week 33 MAPE Score: 0.1229116154110178
     Week 34 MAPE Score: 0.18312786085087188
     Week 35 MAPE Score: 0.23151406338125696
     Week 36 MAPE Score: 0.1461305492763972
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Input In [61], in <cell line: 11>()
         18 model_ARIMA = ARIMA(train, order=(6,1,0))
         19 # Fit Model
    ---> 20 ARIMA_fit = model_ARIMA.fit()
         22 #Evaluate Model
         23 predict = ARIMA_fit.forecast(168)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\tsa\arima\model.py:390, in ARIMA.fit(self, start_params, transformed, includes_fixed, method, method_kwargs, gls, gls_kwargs, cov_type, cov_kwds, return_params, low_memory)
        387 else:
        388     method_kwargs.setdefault('disp', 0)
    --> 390     res = super().fit(
        391         return_params=return_params, low_memory=low_memory,
        392         cov_type=cov_type, cov_kwds=cov_kwds, **method_kwargs)
        393     if not return_params:
        394         res.fit_details = res.mlefit
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\tsa\statespace\mlemodel.py:704, in MLEModel.fit(self, start_params, transformed, includes_fixed, cov_type, cov_kwds, method, maxiter, full_output, disp, callback, return_params, optim_score, optim_complex_step, optim_hessian, flags, low_memory, **kwargs)
        702         flags['hessian_method'] = optim_hessian
        703     fargs = (flags,)
    --> 704     mlefit = super(MLEModel, self).fit(start_params, method=method,
        705                                        fargs=fargs,
        706                                        maxiter=maxiter,
        707                                        full_output=full_output,
        708                                        disp=disp, callback=callback,
        709                                        skip_hessian=True, **kwargs)
        711 # Just return the fitted parameters if requested
        712 if return_params:
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\base\model.py:563, in LikelihoodModel.fit(self, start_params, method, maxiter, full_output, disp, fargs, callback, retall, skip_hessian, **kwargs)
        560     del kwargs["use_t"]
        562 optimizer = Optimizer()
    --> 563 xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
        564                                                fargs, kwargs,
        565                                                hessian=hess,
        566                                                method=method,
        567                                                disp=disp,
        568                                                maxiter=maxiter,
        569                                                callback=callback,
        570                                                retall=retall,
        571                                                full_output=full_output)
        572 # Restore cov_type, cov_kwds and use_t
        573 optim_settings.update(kwds)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\base\optimizer.py:241, in Optimizer._fit(self, objective, gradient, start_params, fargs, kwargs, hessian, method, maxiter, full_output, disp, callback, retall)
        238     fit_funcs.update(extra_fit_funcs)
        240 func = fit_funcs[method]
    --> 241 xopt, retvals = func(objective, gradient, start_params, fargs, kwargs,
        242                      disp=disp, maxiter=maxiter, callback=callback,
        243                      retall=retall, full_output=full_output,
        244                      hess=hessian)
        246 optim_settings = {'optimizer': method, 'start_params': start_params,
        247                   'maxiter': maxiter, 'full_output': full_output,
        248                   'disp': disp, 'fargs': fargs, 'callback': callback,
        249                   'retall': retall, "extra_fit_funcs": extra_fit_funcs}
        250 optim_settings.update(kwargs)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\base\optimizer.py:651, in _fit_lbfgs(f, score, start_params, fargs, kwargs, disp, maxiter, callback, retall, full_output, hess)
        648 elif approx_grad:
        649     func = f
    --> 651 retvals = optimize.fmin_l_bfgs_b(func, start_params, maxiter=maxiter,
        652                                  callback=callback, args=fargs,
        653                                  bounds=bounds, disp=disp,
        654                                  **extra_kwargs)
        656 if full_output:
        657     xopt, fopt, d = retvals
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_lbfgsb_py.py:199, in fmin_l_bfgs_b(func, x0, fprime, args, approx_grad, bounds, m, factr, pgtol, epsilon, iprint, maxfun, maxiter, disp, callback, maxls)
        187     disp = iprint
        188 opts = {'disp': disp,
        189         'iprint': iprint,
        190         'maxcor': m,
       (...)
        196         'callback': callback,
        197         'maxls': maxls}
    --> 199 res = _minimize_lbfgsb(fun, x0, args=args, jac=jac, bounds=bounds,
        200                        **opts)
        201 d = {'grad': res['jac'],
        202      'task': res['message'],
        203      'funcalls': res['nfev'],
        204      'nit': res['nit'],
        205      'warnflag': res['status']}
        206 f = res['fun']
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_lbfgsb_py.py:308, in _minimize_lbfgsb(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, finite_diff_rel_step, **unknown_options)
        305     else:
        306         iprint = disp
    --> 308 sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
        309                               bounds=new_bounds,
        310                               finite_diff_rel_step=finite_diff_rel_step)
        312 func_and_grad = sf.fun_and_grad
        314 fortran_int = _lbfgsb.types.intvar.dtype
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_optimize.py:263, in _prepare_scalar_function(fun, x0, jac, args, bounds, epsilon, finite_diff_rel_step, hess)
        259     bounds = (-np.inf, np.inf)
        261 # ScalarFunction caches. Reuse of fun(x) during grad
        262 # calculation reduces overall function evaluations.
    --> 263 sf = ScalarFunction(fun, x0, args, grad, hess,
        264                     finite_diff_rel_step, bounds, epsilon=epsilon)
        266 return sf
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_differentiable_functions.py:177, in ScalarFunction.__init__(self, fun, x0, args, grad, hess, finite_diff_rel_step, finite_diff_bounds, epsilon)
        173         self.g = approx_derivative(fun_wrapped, self.x, f0=self.f,
        174                                    **finite_diff_options)
        176 self._update_grad_impl = update_grad
    --> 177 self._update_grad()
        179 # Hessian Evaluation
        180 if callable(hess):
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_differentiable_functions.py:256, in ScalarFunction._update_grad(self)
        254 def _update_grad(self):
        255     if not self.g_updated:
    --> 256         self._update_grad_impl()
        257         self.g_updated = True
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_differentiable_functions.py:173, in ScalarFunction.__init__.<locals>.update_grad()
        171 self._update_fun()
        172 self.ngev += 1
    --> 173 self.g = approx_derivative(fun_wrapped, self.x, f0=self.f,
        174                            **finite_diff_options)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_numdiff.py:505, in approx_derivative(fun, x0, method, rel_step, abs_step, f0, bounds, sparsity, as_linear_operator, args, kwargs)
        502     use_one_sided = False
        504 if sparsity is None:
    --> 505     return _dense_difference(fun_wrapped, x0, f0, h,
        506                              use_one_sided, method)
        507 else:
        508     if not issparse(sparsity) and len(sparsity) == 2:
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_numdiff.py:576, in _dense_difference(fun, x0, f0, h, use_one_sided, method)
        574     x = x0 + h_vecs[i]
        575     dx = x[i] - x0[i]  # Recompute dx as exactly representable number.
    --> 576     df = fun(x) - f0
        577 elif method == '3-point' and use_one_sided[i]:
        578     x1 = x0 + h_vecs[i]
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_numdiff.py:456, in approx_derivative.<locals>.fun_wrapped(x)
        455 def fun_wrapped(x):
    --> 456     f = np.atleast_1d(fun(x, *args, **kwargs))
        457     if f.ndim > 1:
        458         raise RuntimeError("`fun` return value has "
        459                            "more than 1 dimension.")
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\scipy\optimize\_differentiable_functions.py:137, in ScalarFunction.__init__.<locals>.fun_wrapped(x)
        133 self.nfev += 1
        134 # Send a copy because the user may overwrite it.
        135 # Overwriting results in undefined behaviour because
        136 # fun(self.x) will change self.x, with the two no longer linked.
    --> 137 fx = fun(np.copy(x), *args)
        138 # Make sure the function returns a true scalar
        139 if not np.isscalar(fx):
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\base\model.py:531, in LikelihoodModel.fit.<locals>.f(params, *args)
        530 def f(params, *args):
    --> 531     return -self.loglike(params, *args) / nobs
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\tsa\statespace\mlemodel.py:939, in MLEModel.loglike(self, params, *args, **kwargs)
        936 if complex_step:
        937     kwargs['inversion_method'] = INVERT_UNIVARIATE | SOLVE_LU
    --> 939 loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
        941 # Koopman, Shephard, and Doornik recommend maximizing the average
        942 # likelihood to avoid scale issues, but the averaging is done
        943 # automatically in the base model `fit` method
        944 return loglike
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\statsmodels\tsa\statespace\kalman_filter.py:987, in KalmanFilter.loglike(self, **kwargs)
        984 loglikelihood_burn = kwargs.get('loglikelihood_burn',
        985                                 self.loglikelihood_burn)
        986 if not (kwargs['conserve_memory'] & MEMORY_NO_LIKELIHOOD):
    --> 987     loglike = np.sum(kfilter.loglikelihood[loglikelihood_burn:])
        988 else:
        989     loglike = np.sum(kfilter.loglikelihood)
    

    File <__array_function__ internals>:180, in sum(*args, **kwargs)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\numpy\core\fromnumeric.py:2298, in sum(a, axis, dtype, out, keepdims, initial, where)
       2295         return out
       2296     return res
    -> 2298 return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
       2299                       initial=initial, where=where)
    

    File ~\anaconda3\envs\timeseries\lib\site-packages\numpy\core\fromnumeric.py:86, in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
         83         else:
         84             return reduction(axis=axis, out=out, **passkwargs)
    ---> 86 return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    

    KeyboardInterrupt: 



```python

```

SARIMAX? Seasonal Arima?


```python

# The test window will step in one week intervals - 168 hours
week_steps = range(n_train, n_records-168, 168)

# MAPE score for each week of the test year will be stored here
Scores = []

# Counter used to track progress of the loop
week = 1

# For loop to test all 52 weeks of the final year of data
for n in week_steps:
    # Train data encompasses the start of the dataset to the start of sliding test set
    train = df_AR[0:n]
    # Test data is 168 hours- or one week long
    test = df_AR[n:n+168]
    
    # Instantiate Model
    model_SARIMAX = SARIMAX(train, order=(6,1,0), seasonal_order=(0,0,0,12))
    # Fit Model
    SARIMAX_fit = model_SARIMAX.fit()
    
    #Evaluate Model
    predict = SARIMAX_fit.forecast(168)
    y_actual = test['total load actual']
    y_predicted = predict.astype(float)
    AR_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    
    print(f' Week {week} MAPE Score: {Prophet_MAPE}')
    Scores.append(AR_MAPE)
    week = week+1

print(f' Combined MAPE Score: {sum(Scores)/len(Scores)}')
```

### Plotting


```python
y_remainder = y_predicted.subtract(y_actual)
```


```python
y_remainder.head()
```


```python
plt.figure(figsize=(11,6))

# Left and Right ends of the plot line up with the test set
left = df_AR.index[n:n+1]
right = df_AR.index[n+168:n+169]


# These are the forecasted values for the next 100 hours
plt.plot(test.index, y_actual, color='crimson', label='Total Load Actual')

plt.plot(test.index, y_predicted, color='cornflowerblue', label='Total Load Predicted')

plt.plot(test.index, y_remainder, color='green', label='Remainder')

plt.xlim([left, right])

plt.axhline(color='yellow', label='Zero Line')

plt.axhline(min(y_remainder), color='black', label= f'1 Week Minimum: {min(y_remainder)}')

plt.legend()

plt.ylabel('Total Electrical Load (MW)')

plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# Auto Regressive Model

### Dataframe preparation


```python
df_AR = pd.DataFrame(df_e3['total load actual'])
df_AR
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
      <th>total load actual</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-12-31 23:00:00+00:00</th>
      <td>25385.0</td>
    </tr>
    <tr>
      <th>2015-01-01 00:00:00+00:00</th>
      <td>24382.0</td>
    </tr>
    <tr>
      <th>2015-01-01 01:00:00+00:00</th>
      <td>22734.0</td>
    </tr>
    <tr>
      <th>2015-01-01 02:00:00+00:00</th>
      <td>21286.0</td>
    </tr>
    <tr>
      <th>2015-01-01 03:00:00+00:00</th>
      <td>20264.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-31 18:00:00+00:00</th>
      <td>30653.0</td>
    </tr>
    <tr>
      <th>2018-12-31 19:00:00+00:00</th>
      <td>29735.0</td>
    </tr>
    <tr>
      <th>2018-12-31 20:00:00+00:00</th>
      <td>28071.0</td>
    </tr>
    <tr>
      <th>2018-12-31 21:00:00+00:00</th>
      <td>25801.0</td>
    </tr>
    <tr>
      <th>2018-12-31 22:00:00+00:00</th>
      <td>24455.0</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 1 columns</p>
</div>




```python
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

### Persistence (Baseline) Model


```python
# For now, 3 years train, and one week test
train = df_AR[0:26306]
test = df_AR[26306:26474]
```


```python
y_actual_baseline = test['total load actual']
```


```python
# Most recent load value from the training set used indefinately as the prediction
y_pred_baseline = np.full(len(test), train[-1:])
```


```python
# Testing the baseline Mean Absolute Percantage Error
Baseline_MAPE = mean_absolute_percentage_error(y_actual_baseline, y_pred_baseline)
Baseline_MAPE
```




    0.18117331804451783



Our baseline model acheives 18.12% MAPE on the first week of test data. 18% will serve as our baseline MAPE: any model built after this point will need to perform better to be considered an upgrade over the baseline model.

Let's visualize the performance of the baseline


```python
# Plotting the AR model
plt.figure(figsize=(11,6))
# Left and Right ends of the plot line up with the test set
left = test.index[:1]
right = test.index[-1:]
# Plotting
plt.plot(test.index, y_actual_baseline, color='gold', label='Total Load Actual')
plt.plot(test.index, y_pred_baseline, color='cornflowerblue', label='Total Load Predicted (Baseline Model)')
plt.xlim([left, right])
plt.legend()
plt.ylabel('Total Electrical Load (MW)')
plt.show()
```


    
![png](output_131_0.png)
    


Unsurprisingly, the performance of the baseline model is not great as it remains completely stationary, albeit in the general vicinity of the actual load. Since the baseline uses only the most recent load value to make predictions, in this case a sunday night from 11pm to midnight, it is underpredicting load values for most of the following week- save for a few hours late at night.

### AR Model

After the baseline model, we can try to fit an autoregressive model to the load values. As a starting point, i've picked a lag value of 50, or just over two days of past data.


```python
# Instantiate and fit model
AR_Model = AutoReg(train, lags=50).fit()
```


```python
# Make predictions, test predictions against actual values and calculate MAPE
predict = AR_Model.forecast(168)
y_actual = test['total load actual']
y_predicted = predict.astype(float)
AR_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
AR_MAPE
```




    0.15212064578517914



With the guess value of 50 lags, the AR model does improve slightly on the performance of the baseline, reaching 15.21% MAPE

To visualize the performance of the AutoRegression Model, we plot the model alongside the actual results


```python
# Plotting the AR model
plt.figure(figsize=(11,6))
# Left and Right ends of the plot line up with the test set
left = test.index[:1]
right = test.index[-1:]
# Plotting
plt.plot(test.index, y_actual, color='gold', label='Total Load Actual')
plt.plot(test.index, y_predicted, color='cornflowerblue', label='Total Load Predicted (AR Model)')
plt.plot(test.index, y_pred_baseline, color='crimson', label='Total Load Predicted (Baseline Model)')

plt.xlim([left, right])
plt.legend()
plt.ylabel('Total Electrical Load (MW)')
plt.show()
```


    
![png](output_139_0.png)
    


With 50 lags, the auto-regressive model clearly picks up on the day to day patterns of load: two peaks in the morning and evening, with a significant low during the night. The model also picks up on a slight upward trend between days 1 and 2, but large residuals remain. Much of the patterns in load seen this week can be explained by two holidays: New Year's Day and Epiphany. Many Spaniards were off work on the 1st, 4th, and 5th, explaining the lower load values.

### Parameter Optimization:

While 50 lags (just over two days) performed significantly better than our baseline, there is likely another lag value that performs better. We can run a loop to build and compare models with different lag values.
While an ideal model would be able to account for holidays and other irregularities in the schedules of Spaniards, this is likely out of the AR model's scope. For optimizing hyperparameters, it is probably best that the model is tested on a more typical workweek. We will set the weeklong 'test' window on the third week of the January to give it some space from holiday disruptions.


```python
# Test is now on the second week of year #4. Train still goes up to the start of test
train = df_AR[0:26642]
test = df_AR[26642:26810]
```


```python
# I'll test lag values for up to the previous month (Approx 730 horus in a month)
lag_range = np.arange(0,731,5)
```


```python
# Building a model for each lag value in 'lag_range'. The model scores are collected and will be assessed for the
# optimal lag value
test_scores = []
for lag in lag_range:
    AR_Model = AutoReg(train, lags=lag).fit()
    predict = AR_Model.forecast(168)
    y_actual = test['total load actual']
    y_predicted = predict.astype(float)
    AR_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    print(f'Lag Value: {lag}  MAPE: {AR_MAPE}')
    test_scores.append(AR_MAPE)
    
```

    Lag Value: 0  MAPE: 0.1519021870320439
    Lag Value: 5  MAPE: 0.14783530756002744
    Lag Value: 10  MAPE: 0.1481253992835272
    Lag Value: 15  MAPE: 0.14621483318416365
    Lag Value: 20  MAPE: 0.14406491683552017
    Lag Value: 25  MAPE: 0.11492322304120263
    Lag Value: 30  MAPE: 0.11493594218898925
    Lag Value: 35  MAPE: 0.11131973604177939
    Lag Value: 40  MAPE: 0.11099934126824258
    Lag Value: 45  MAPE: 0.11118430843440211
    Lag Value: 50  MAPE: 0.11082868436198808
    Lag Value: 55  MAPE: 0.11092558984004007
    Lag Value: 60  MAPE: 0.1113917356715225
    Lag Value: 65  MAPE: 0.11147677177806392
    Lag Value: 70  MAPE: 0.11173626482715808
    Lag Value: 75  MAPE: 0.10992216541203648
    Lag Value: 80  MAPE: 0.11004185920700756
    Lag Value: 85  MAPE: 0.1111953987973205
    Lag Value: 90  MAPE: 0.11184532274448249
    Lag Value: 95  MAPE: 0.11215656228835501
    Lag Value: 100  MAPE: 0.11007887758912661
    Lag Value: 105  MAPE: 0.10999161022592029
    Lag Value: 110  MAPE: 0.10973240093929826
    Lag Value: 115  MAPE: 0.10953715580777153
    Lag Value: 120  MAPE: 0.10955708825619564
    Lag Value: 125  MAPE: 0.10926901405375787
    Lag Value: 130  MAPE: 0.10933405665385551
    Lag Value: 135  MAPE: 0.10975296561046696
    Lag Value: 140  MAPE: 0.11172700918997427
    Lag Value: 145  MAPE: 0.11123946236963758
    Lag Value: 150  MAPE: 0.11140079893568633
    Lag Value: 155  MAPE: 0.11100598134361632
    Lag Value: 160  MAPE: 0.11110443476557479
    Lag Value: 165  MAPE: 0.11151257452618338
    Lag Value: 170  MAPE: 0.11105062573304518
    Lag Value: 175  MAPE: 0.1111664370067908
    Lag Value: 180  MAPE: 0.11155866116217741
    Lag Value: 185  MAPE: 0.11164130784719403
    Lag Value: 190  MAPE: 0.11168633589185573
    Lag Value: 195  MAPE: 0.11174867003857909
    Lag Value: 200  MAPE: 0.11179793181790784
    Lag Value: 205  MAPE: 0.11161870687974842
    Lag Value: 210  MAPE: 0.11141322903532251
    Lag Value: 215  MAPE: 0.1112926942760388
    Lag Value: 220  MAPE: 0.1112858681660223
    Lag Value: 225  MAPE: 0.11131125684040175
    Lag Value: 230  MAPE: 0.11124227615614021
    Lag Value: 235  MAPE: 0.11114186262648518
    Lag Value: 240  MAPE: 0.11077668721446648
    Lag Value: 245  MAPE: 0.10961667081041733
    Lag Value: 250  MAPE: 0.11014729552764982
    Lag Value: 255  MAPE: 0.10994750149179551
    Lag Value: 260  MAPE: 0.1097294846051826
    Lag Value: 265  MAPE: 0.10658919070959663
    Lag Value: 270  MAPE: 0.10588244611161744
    Lag Value: 275  MAPE: 0.1059157774984337
    Lag Value: 280  MAPE: 0.10426879549506614
    Lag Value: 285  MAPE: 0.1032849201418404
    Lag Value: 290  MAPE: 0.10269329204316746
    Lag Value: 295  MAPE: 0.10222733513378086
    Lag Value: 300  MAPE: 0.10259788117083579
    Lag Value: 305  MAPE: 0.10252423252892719
    Lag Value: 310  MAPE: 0.10200000791613849
    Lag Value: 315  MAPE: 0.10319715956399116
    Lag Value: 320  MAPE: 0.10308694052658933
    Lag Value: 325  MAPE: 0.1031188186101347
    Lag Value: 330  MAPE: 0.10323150692576874
    Lag Value: 335  MAPE: 0.10354497829449102
    Lag Value: 340  MAPE: 0.10299827042553457
    Lag Value: 345  MAPE: 0.10309936802396669
    Lag Value: 350  MAPE: 0.1028735954196433
    Lag Value: 355  MAPE: 0.10322411675444625
    Lag Value: 360  MAPE: 0.10352895513344293
    Lag Value: 365  MAPE: 0.10346865286562573
    Lag Value: 370  MAPE: 0.10384945110905848
    Lag Value: 375  MAPE: 0.10390534753211031
    Lag Value: 380  MAPE: 0.10386937075381855
    Lag Value: 385  MAPE: 0.10360784905271687
    Lag Value: 390  MAPE: 0.10357429746055648
    Lag Value: 395  MAPE: 0.10382892679953
    Lag Value: 400  MAPE: 0.10305299961275723
    Lag Value: 405  MAPE: 0.10367740608146696
    Lag Value: 410  MAPE: 0.10225691181346756
    Lag Value: 415  MAPE: 0.10163205671932737
    Lag Value: 420  MAPE: 0.10316472445877263
    Lag Value: 425  MAPE: 0.10183396348208876
    Lag Value: 430  MAPE: 0.10198022196711255
    Lag Value: 435  MAPE: 0.10114841014317137
    Lag Value: 440  MAPE: 0.10153339049324724
    Lag Value: 445  MAPE: 0.10185728705777367
    Lag Value: 450  MAPE: 0.10209884330243771
    Lag Value: 455  MAPE: 0.10176010506579074
    Lag Value: 460  MAPE: 0.10173785667856716
    Lag Value: 465  MAPE: 0.10136011023725355
    Lag Value: 470  MAPE: 0.10033639134171751
    Lag Value: 475  MAPE: 0.09911477047517192
    Lag Value: 480  MAPE: 0.09895474092483832
    Lag Value: 485  MAPE: 0.09924323903623393
    Lag Value: 490  MAPE: 0.09936805704910347
    Lag Value: 495  MAPE: 0.09951287458324505
    Lag Value: 500  MAPE: 0.09971081830851389
    Lag Value: 505  MAPE: 0.09927172036575732
    Lag Value: 510  MAPE: 0.09939555746501946
    Lag Value: 515  MAPE: 0.09971446421007729
    Lag Value: 520  MAPE: 0.09988048054199708
    Lag Value: 525  MAPE: 0.09824051220908665
    Lag Value: 530  MAPE: 0.09879641538104075
    Lag Value: 535  MAPE: 0.09865096265782469
    Lag Value: 540  MAPE: 0.09797802087866866
    Lag Value: 545  MAPE: 0.09815196607355782
    Lag Value: 550  MAPE: 0.09826607570795247
    Lag Value: 555  MAPE: 0.09762716564157056
    Lag Value: 560  MAPE: 0.09750846333042983
    Lag Value: 565  MAPE: 0.09757251319897617
    Lag Value: 570  MAPE: 0.09768054445005336
    Lag Value: 575  MAPE: 0.09802122078804631
    Lag Value: 580  MAPE: 0.09786212213951147
    Lag Value: 585  MAPE: 0.0980540607730605
    Lag Value: 590  MAPE: 0.0981845493358494
    Lag Value: 595  MAPE: 0.09834803302336789
    Lag Value: 600  MAPE: 0.09829204228542993
    Lag Value: 605  MAPE: 0.09902130927161444
    Lag Value: 610  MAPE: 0.10051092193229144
    Lag Value: 615  MAPE: 0.10345006176606264
    Lag Value: 620  MAPE: 0.10265866487271046
    Lag Value: 625  MAPE: 0.10262666151270701
    Lag Value: 630  MAPE: 0.10059617030386321
    Lag Value: 635  MAPE: 0.09142474305026188
    Lag Value: 640  MAPE: 0.09450908767620772
    Lag Value: 645  MAPE: 0.08444919552761476
    Lag Value: 650  MAPE: 0.0926406371378367
    Lag Value: 655  MAPE: 0.08749362491913172
    Lag Value: 660  MAPE: 0.07300611602693667
    Lag Value: 665  MAPE: 0.07061533122771052
    Lag Value: 670  MAPE: 0.062319895141205534
    Lag Value: 675  MAPE: 0.0760848485963071
    Lag Value: 680  MAPE: 0.07539912632186505
    Lag Value: 685  MAPE: 0.07430245623246332
    Lag Value: 690  MAPE: 0.07416231083167009
    Lag Value: 695  MAPE: 0.07230127602770989
    Lag Value: 700  MAPE: 0.0712210657239815
    Lag Value: 705  MAPE: 0.0703220483944654
    Lag Value: 710  MAPE: 0.07025276077708112
    Lag Value: 715  MAPE: 0.07027679558371532
    Lag Value: 720  MAPE: 0.0702806602034889
    Lag Value: 725  MAPE: 0.07015664170857118
    Lag Value: 730  MAPE: 0.07036804281627684
    


```python
plt.figure()

plt.plot(test_scores)
plt.xlabel('Lag Increment')
plt.ylabel('Mean Absolute Percentage Error')
plt.xticks(ticks=[0,20,40,60,80,100,120,140], labels=[0,100,200,300,400,500,600,700])
plt.title('AutoRegressive Model: Optimization of Lag Increment')
plt.show()
```


    
![png](output_145_0.png)
    



```python
min(test_scores)
```

The best MAPE score over this particular week comes from a lag increment of 435. For now, we'll accept this as the optimal model parameter.

### Rerun of the Auto Regressive Model with a Lag of 435


```python
AR_Model = AutoReg(train, lags=435).fit()
```


```python
predict = AR_Model.forecast(168)
y_actual = test['total load actual']
y_predicted = predict.astype(float)
# Since we changed train and test, we need to reset the baseline model for the new test week
y_pred_baseline = np.full(len(test), train[-1:])
AR_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
AR_MAPE
```


```python
# Plotting the AR model
plt.figure(figsize=(11,6))
# Left and Right ends of the plot line up with the test set
left = test.index[:1]
right = test.index[-1:]
# Plotting
plt.plot(test.index, y_actual, color='gold', label='Total Load Actual')
plt.plot(test.index, y_predicted, color='cornflowerblue', label='Total Load Predicted (AR Model)')
plt.plot(test.index, y_pred_baseline, color='crimson', label='Total Load Predicted (Baseline Model)')

plt.xlim([left, right])
plt.legend()
plt.ylabel('Total Electrical Load (MW)')
plt.show()
```

### A Better Way to Test

So far, we have only been evaluating models based on their performance in either the first, or second week of 2018. While I'd like to stick to my evaluation criteria of Mean Absolute Percentage Error over one week (168 predecitions), it would be best to test over every week of 2018.
In order to accomplish this, I'll impliment a sliding test window so that 52 models can be run over the course of 2018, and their scores averaged. The train window will expand as the test window slides, so that every prediction is made for the one week of time immediately following the training set.


```python
# Using a rolling dataframe!
# Dataframe formatted for AR model
df_AR

# Number of records in training set. This is the first three years of entries, and will be included in every train set
n_train = 26306 # 3 Years, Test set starts 1/1/2018 at midnight (first datapoint at 1AM), which also happens to be a monday
n_records = len(df_AR)
```


```python
# The test window will step in one week intervals - 168 hours
week_steps = range(n_train, n_records-168, 168)

# MAPE score for each week of the test year will be stored here
AR_Scores = []
Base_Scores = []

# Counter used to track progress of the loop
week = 1

# For loop to test all 52 weeks of the final year of data
for n in week_steps:
    # Train data encompasses the start of the dataset to the start of sliding test set
    train = df_AR[0:n]
    # Test data is 168 hours- or one week long
    test = df_AR[n:n+168]
    
    # Instantiate + Fit Model
    AR_Model = AutoReg(train, lags=435).fit()
    
    #Evaluate Model
    predict = AR_Model.forecast(168)
    y_actual = test['total load actual']
    y_predicted = predict.astype(float)
    # Since we keep changing train and test, we need to rebuilt the baseline model for each new test week
    y_pred_baseline = np.full(len(test), train[-1:])
    AR_MAPE = mean_absolute_percentage_error(y_actual, y_predicted)
    Base_MAPE = mean_absolute_percentage_error(y_actual, y_pred_baseline)
    
    print(f' Week {week} MAPE Score (Auto Resressive Model): {AR_MAPE}')
    Scores.append(AR_Scores)
    
    print(f' Week {week} MAPE Score (Baseline Model): {Base_MAPE}')
    Scores.append(Base_Scores)
    
    week = week+1

print(f' Combined MAPE Score: {sum(Scores)/len(Scores)}')
```


```python

```
