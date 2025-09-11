import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

import tensorflow as tf
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        """
        Load the EfficientNet base model with pre-trained ImageNet weights.
        """
        self.model = tf.keras.applications.efficientnet.EfficientNetB0(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,      # e.g., 'imagenet'
            include_top=self.config.params_include_top  # False for transfer learning
        )

        # Save the base model before adding custom layers
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model by adding a classification head to the EfficientNet base.
        """
        # Freeze layers if required
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dropout(0.4)(x)  # Dropout to prevent overfitting
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Add the classification head to EfficientNet and save the updated model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,  # Freeze entire base for initial training
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the model to a given path.
        """
        model.save(path)
