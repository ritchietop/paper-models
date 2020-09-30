import tensorflow as tf
from absl import app, flags
from paper_models.models.lr import LRModel
from examples.data.titanic_data import get_dataset


flags.DEFINE_string("model_name", "LR", "the name of model.")
flags.DEFINE_integer("units", 1, "")
flags.DEFINE_float("learning_rate", 0.05, "")
FLAGS = flags.FLAGS


def test(_):
    train_data, test_data = get_dataset(batch_size=200)
    pclass_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup()
    pclass_input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="Pclass")
    pclass_input = pclass_layer(pclass_input_layer)
    pclass_input = tf.keras.layers.Embedding(input_dim=5, output_dim=10)(pclass_input)

    lr_model = LRModel(units=FLAGS.units)
    logits = lr_model(pclass_input)

    model = tf.keras.Model(inputs=[pclass_input_layer], outputs=[logits])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
                  loss=tf.keras.losses.binary_crossentropy)
    model.fit(x=train_data, epochs=10)


if __name__ == "__main__":
    app.run(test)
