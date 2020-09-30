import tensorflow as tf
from tensorflow.python.feature_column.feature_column_lib import FeatureColumn, CategoricalColumn, DenseColumn
from tensorflow.python.feature_column.feature_column_v2 import _StateManagerImplV2, FeatureTransformationCache
from typing import List


class LRLayer(tf.keras.layers.Layer):
    def __init__(self, units, columns: List[FeatureColumn], combiner="sum", trainable=True, name=None, dtype=tf.float32,
                 dynamic=False, **kwargs):
        super(LRLayer, self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.units = units
        self.columns = sorted(columns, key=lambda column: column.name)
        self.combiner = combiner
        self.state_manager = _StateManagerImplV2(self, self.trainable)
        self.bias = None

    def build(self, input_shape):
        for column in self.columns:
            with tf.name_scope("init_weights[%s]" % column.name):
                column.create_state(self.state_manager)
                if isinstance(column, CategoricalColumn):
                    dimension = column.num_buckets
                elif isinstance(column, DenseColumn):
                    dimension = column.variable_shape.num_elements()
                else:
                    raise ValueError("Only support CategoricalColumn and DenseColumn. But got {}.".format(type(column)))
                self.state_manager.create_variable(
                    feature_column=column,
                    name="weights",
                    shape=(dimension, self.units),
                    dtype=self.dtype,
                    trainable=self.trainable)
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            dtype=self.dtype,
            trainable=self.trainable,
            initializer=tf.keras.initializers.Zeros())

    def call(self, inputs, training=None, mask=None):
        transformation_cache = FeatureTransformationCache(inputs)
        output_tensors = []
        for column in self.columns:
            with tf.name_scope(column.name):
                column_weights = self.state_manager.get_variable(column, "weights")
                if isinstance(column, CategoricalColumn):
                    sparse_tensors = column.get_sparse_tensors(transformation_cache, self.state_manager)
                    output_tensor = tf.nn.safe_embedding_lookup_sparse(
                        embedding_weights=column_weights,
                        sparse_ids=sparse_tensors.id_tensor,
                        sparse_weights=sparse_tensors.weight_tensor,
                        combiner=self.combiner)  # batch_size * units
                elif isinstance(column, DenseColumn):
                    dense_tensor = column.get_dense_tensor(transformation_cache, self.state_manager)
                    output_tensor = tf.matmul(dense_tensor, column_weights)  # batch_size * units
                else:
                    raise ValueError("Only support CategoricalColumn and DenseColumn. But got {}.".format(type(column)))
                output_tensors.append(output_tensor)
        predictions_no_bias = tf.math.accumulate_n(output_tensors)
        predictions = tf.nn.bias_add(predictions_no_bias, self.bias)
        return predictions

    def get_config(self):
        from tensorflow.python.feature_column.serialization import serialize_feature_columns
        config = {
            "units": self.units,
            "columns": serialize_feature_columns(self.columns),
            "combiner": self.combiner
        }
        base_config = super(LRLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.feature_column.serialization import deserialize_feature_columns
        config_cp = config.copy()
        config_cp["columns"] = deserialize_feature_columns(config["columns"], custom_objects=custom_objects)
        return cls(**config_cp)
