import tensorflow as tf


def get_dataset(batch_size):
    train_path = "./data/titanic/train.csv"
    train_column_names = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                          "Cabin", "Embarked"]
    train_column_defaults = ["", 0, 0, "", "", 0.0, -1, -1, "", 0.0, "", ""]
    train_data = tf.data.experimental.make_csv_dataset(file_pattern=train_path, batch_size=batch_size, num_epochs=1,
                                                       column_names=train_column_names,
                                                       column_defaults=train_column_defaults, label_name="Survived",
                                                       num_rows_for_inference=0)

    test_path = "./data/titanic/test.csv"
    test_column_names = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                         "Cabin", "Embarked"]
    test_column_defaults = ["", 0, "", "", 0.0, -1, -1, "", 0.0, "", ""]
    test_data = tf.data.experimental.make_csv_dataset(file_pattern=test_path, batch_size=batch_size, num_epochs=1,
                                                      column_names=test_column_names,
                                                      column_defaults=test_column_defaults,
                                                      num_rows_for_inference=0)

    return train_data, test_data


def feature_processing(features):
    pclass_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=[1, 2, 3], num_oov_indices=0)
    pclass_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="Pclass")
    sex_process = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=['male', 'female'],
                                                                          num_oov_indices=0)
    sex_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Sex")
    age_process = tf.keras.layers.experimental.preprocessing.Discretization(bins=[6, 12, 18, 30, 40, 50, 60])
    age_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="Age")
    sibsp_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(9)),
                                                                             num_oov_indices=0)
    sibsp_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="SibSp")
    parch_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(7)),
                                                                             num_oov_indices=0)
    parch_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="Parch")
    ticket_process = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
    ticket_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Ticket")
    fare_process = tf.keras.layers.Lambda(function=lambda tensor: tf.math.log1p(tensor))
    fare_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="Fare")
    cabin_process = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
    cabin_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Cabin")
    embarked_process = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["S", "C"], num_oov_indices=0)
    embarked_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Embarked")

from sklearn.preprocessing import OrdinalEncoder


