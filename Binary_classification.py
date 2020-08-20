import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras import layers
from tensorflow import feature_column
from matplotlib import pyplot as plt

# settings display options
pd.options.display.float_format = "{:.2f}".format

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# shuffle the data

train_df = train_df.reindex(np.random.permutation(train_df.index))

print("RAW DATA")
print(train_df.head())

# lets normalize using z-value
# z-value is (value - mean) / standard_deviation

train_df_norm = (train_df - train_df.mean()) / train_df.std()

print("z-normalized data")
print(train_df_norm.head(10))

# apply on test data too
test_df_norm = (test_df - test_df.mean()) / test_df.std()


# creating a binary label
threshold = 265000

train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold)

print(train_df_norm["median_house_value_is_high"].head(10))


