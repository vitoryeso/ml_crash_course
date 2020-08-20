import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib

from simple_colors import *
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

pd.options.display.float_format = "{:.2f}".format

dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.')

dataframe = pd.read_csv(csv_file)

# in this dataset, 4 value for adoptionSpeed indicates the pet was not adopted 
# np.where is fasther them tf.where, when you're on CPU
dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)

dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    targets = dataframe.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    return dataset

# splitting
train_df, test_df = train_test_split(dataframe, test_size=0.2)
train_df, val_df = train_test_split(dataframe, test_size=0.2)

batch_size = 5

# creating datasets
train_ds = df_to_dataset(train_df, batch_size=batch_size)
test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)
val_ds = df_to_dataset(val_df, shuffle=False, batch_size=batch_size)

example = next(iter(train_ds))[0]

def demo(feature_column):
    layer = layers.DenseFeatures(feature_column)
    return layer(example)

categorical = ['Type', 'Color1', 'Color2', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health'] 
embedded = ['Breed1']
numerical = ['PhotoAmt', 'Age', 'Fee']

feature_columns = []
print(dataframe.columns)

print(categorical)
for categ in categorical:
    fc_categ = feature_column.categorical_column_with_vocabulary_list(categ, dataframe[categ].unique())
    fc_onehot = feature_column.indicator_column(fc_categ)
    feature_columns.append(fc_onehot)
    print(demo(feature_columns[-1]))

# first we create a numeric feature columns and after if we want, we can create bucketized colums

print(numerical)
for num in numerical:
    fc_num = feature_column.numeric_column(num)
    feature_columns.append(fc_num)
    print(demo(feature_columns[-1]))





