
# Your mission

This time, you will build a basic Neural Network model to predict Bitcoin price based on historical Data.
This notebook helps you to train a model but you can use it however you want.


```python
import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

import helper_functions as hf

CURDIR = os.path.dirname(os.getcwd())
DATADIR = os.path.join(CURDIR,  "data")
FIGDIR = os.path.join(CURDIR,  "figure")
%matplotlib inline
```

    Using TensorFlow backend.


# Import Data
Our Data come from https://blockchain.info/.

Here, we load data into a Pandas DataFrame


```python
df_blockchain = pd.read_csv(os.path.join(DATADIR, "df_blockchain.csv"), delimiter=";")
```


```python
df_blockchain.head()
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
      <th>market-price</th>
      <th>n-transactions-per-block</th>
      <th>median-confirmation-time</th>
      <th>hash-rate</th>
      <th>difficulty</th>
      <th>miners-revenue</th>
      <th>trade-volume</th>
      <th>blocks-size</th>
      <th>avg-block-size</th>
      <th>...</th>
      <th>cost-per-transaction</th>
      <th>n-unique-addresses</th>
      <th>n-transactions</th>
      <th>n-transactions-total</th>
      <th>n-transactions-excluding-popular</th>
      <th>output-volume</th>
      <th>estimated-transaction-volume</th>
      <th>estimated-transaction-volume-usd</th>
      <th>total-bitcoins</th>
      <th>market-cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-09-13</td>
      <td>6.88</td>
      <td>45.908451</td>
      <td>0.0</td>
      <td>12.018816</td>
      <td>1.777774e+06</td>
      <td>52318.011503</td>
      <td>0.0</td>
      <td>592.190091</td>
      <td>0.019009</td>
      <td>...</td>
      <td>7.666766</td>
      <td>12622.0</td>
      <td>6519.0</td>
      <td>1497195.0</td>
      <td>6519.0</td>
      <td>358543.612114</td>
      <td>58615.641320</td>
      <td>403275.612279</td>
      <td>7.257416e+06</td>
      <td>5.022014e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-09-14</td>
      <td>6.19</td>
      <td>42.465753</td>
      <td>0.0</td>
      <td>13.263925</td>
      <td>1.755425e+06</td>
      <td>48306.468911</td>
      <td>0.0</td>
      <td>594.907367</td>
      <td>0.018007</td>
      <td>...</td>
      <td>7.369408</td>
      <td>12408.0</td>
      <td>6200.0</td>
      <td>1503780.0</td>
      <td>6200.0</td>
      <td>302619.024544</td>
      <td>74521.484625</td>
      <td>461287.989830</td>
      <td>7.264662e+06</td>
      <td>4.540930e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-09-15</td>
      <td>5.92</td>
      <td>41.500000</td>
      <td>0.0</td>
      <td>12.914875</td>
      <td>1.755425e+06</td>
      <td>60431.444952</td>
      <td>0.0</td>
      <td>597.554226</td>
      <td>0.018240</td>
      <td>...</td>
      <td>7.333913</td>
      <td>12988.0</td>
      <td>6474.0</td>
      <td>1509972.0</td>
      <td>6474.0</td>
      <td>299226.130646</td>
      <td>79422.402932</td>
      <td>470180.625359</td>
      <td>7.272284e+06</td>
      <td>4.322228e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-09-16</td>
      <td>5.58</td>
      <td>52.176471</td>
      <td>0.0</td>
      <td>10.995096</td>
      <td>1.755425e+06</td>
      <td>34345.021913</td>
      <td>0.0</td>
      <td>600.362512</td>
      <td>0.022136</td>
      <td>...</td>
      <td>5.466341</td>
      <td>12059.0</td>
      <td>6209.0</td>
      <td>1516381.0</td>
      <td>6209.0</td>
      <td>674606.861338</td>
      <td>82696.853247</td>
      <td>461448.441118</td>
      <td>7.279040e+06</td>
      <td>4.088136e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-09-17</td>
      <td>5.18</td>
      <td>40.701493</td>
      <td>0.0</td>
      <td>10.733308</td>
      <td>1.755425e+06</td>
      <td>36805.913687</td>
      <td>0.0</td>
      <td>602.995510</td>
      <td>0.017116</td>
      <td>...</td>
      <td>6.489054</td>
      <td>10988.0</td>
      <td>5454.0</td>
      <td>1522600.0</td>
      <td>5454.0</td>
      <td>354198.945778</td>
      <td>68238.166521</td>
      <td>353473.702578</td>
      <td>7.285375e+06</td>
      <td>3.801833e+07</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



# Explore Dataset

We already Explore dataset before, but you can draw other plots to analyse data if you want.

Idea : you can use pandas_profiling 

```python
from pandas_profiling import ProfileReport
ProfileReport(df)
```


```python
# Your Code Here
```


```python
# get columns (You can add more columns to analyse results)
columns = ["market-price"]
dataset = df_blockchain[columns]
```

# Data scaling

Here we scale price between 0 and 1, this well help the optimization algorithm converge faster.

See the following figure (source : Andrew Ng https://www.andrewng.org ) :

![alt text](../data/feature-scaling.png "Title")


```python
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1,len(columns)))
```


```python
dataset.shape
```

# Process data (Making sequences)

here we process data before training.

LSTM layer as an input layer expects the data to be 3 dimensions, we will use 'process_data' function to split data into sequences of a fixed length (rnn_size).

The neural network is expecting to have an input's shap of [batch_size, rnn_size, nb_features]


```python
def process_data(data, rnn_size=rnn_size, target_id=0, columns_size=len(columns)):
    X = []
    y = []
    for i in range(len(data)-rnn_size):
        X.append(data[i:i+rnn_size,:])
        y.append(data[i+rnn_size,0])
    return np.array(X).astype(np.float32).reshape((-1,rnn_size,columns_size)), np.array(y).astype(np.float32)
```


```python
# process data for RNN
X_train, y_train = process_data(data_train)
X_val, y_val = process_data(data_valid)
X_test, y_test = process_data(data_test)
```


```python
X_train.shape
```


```python
X_val.shape
```


```python
X_test.shape
```

# Deep Learning Model

Here we initialize the model using Keras.

Here we propose to code a basic neural network LSTM + Dense, but you are free to use any architecture.


```python
# neural network model

# Initialising the model
regressor = Sequential()

# Adding the input/LSTM layer 
#Your Code Here
regressor.add()

# Adding the output layer
#Your Code Here
regressor.add()

#Compiling the Recurrent Neural Network with adam optimier and 'mean_absolute_error' as loss function
regressor.compile()
```


```python
#Fitting the Recurrent Neural Network
regressor.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 32, epochs = 50)
```

# Deep Learning Model

Here we initialize the model using Keras.

Here we propose to code a basic neural network LSTM + Dense, but you are free to use any architecture.


```python
# neural network model

# Initialising the model
regressor = Sequential()

# Adding the input/LSTM layer 
#Your Code Here
regressor.add()

# Adding the output layer
#Your Code Here
regressor.add()

#Compiling the Recurrent Neural Network with adam optimier and 'mean_absolute_error' as loss function
regressor.compile()
```

# Evaluation


```python
# compute prediction for test
y_pred = 
```


```python
# compute rmse for test
y_pred_inverse = scaler.inverse_transform(np.concatenate([y_pred, data_test[-len(y_pred):,1:]], axis=1))
y_test_inverse = scaler.inverse_transform(data_test.reshape(-1,len(columns)))[rnn_size:]

rmse_score = 
print("rmse score : ", rmse_score)
```


```python
#Graphs for predicted values
plt.plot(y_test_inverse[rnn_size:,0], color = 'red', label = 'true BTC price')
plt.plot(y_pred_inverse[:,0], color = 'blue', label = 'predicted BTC price')
plt.title('BTC price Prediction')
plt.xlabel('Days')
plt.ylabel('BTC price')
plt.legend()
plt.show()
```


```python
# If you get this far, you can : 
- Test other neural network models
- Test other optimizers
- Compare result between Arima and RNN models
- Find a way to choose most important variables
- ...
```
