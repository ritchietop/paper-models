import tensorflow as tf


class LRModel(tf.keras.Model):
    def __init__(self, units, name=None, **kwargs):
        super(LRModel, self).__init__(name=name, **kwargs)
        self.units = units
        self.linear_layer = tf.keras.layers.Dense(units=units)

    def call(self, inputs, training=None, mask=None):
        return self.linear_layer(inputs)

    def get_config(self):
        config = {"units": self.units}
        config_base = super(LRModel, self).get_config()
        return dict(list(config.items()) + list(config_base.items()))
