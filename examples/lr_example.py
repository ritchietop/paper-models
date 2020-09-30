import tensorflow as tf
from absl import app, flags
from examples.data import TitanicData

data = TitanicData.input_fn()
for features, label in data:
    print(features)
    print(label)
    break
