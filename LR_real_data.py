import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

training_df["median_house_value"] /= 1000.0

plt.matshow(training_df.corr())
plt.show()
