import tensorflow as tf
import os
import logging

from tensorflow.python.keras.backend import flatten


def get_VGG_16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"VGG16 base model saved at: {model_path}")
    return model

def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:freeze_till]:
            layer.trainable = False
        for layers in model.layers[freeze_till :]:
            layer.trainable = True

    ## add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    logging.info("custom model is compiled and ready to be trained. \n")

    return full_model


def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from {untrained_full_model_path} ....")
    return model
