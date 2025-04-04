import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import uuid 
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
    

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=(*self.config.params_image_size,3),
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(base_model, classes, freeze_all, freeze_till, learning_rate):
        
    
        # Generate a unique prefix for layer names
        unique_prefix = str(uuid.uuid4())[:8]  
        
        if freeze_all:
            for layer in base_model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in base_model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten(name=f"custom_flatten_{unique_prefix}")(base_model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax',
            name=f"custom_dense_{unique_prefix}"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
        

    def update_base_model(self):
        if self.model is None:
            raise ValueError("Base model not initialized. Call get_base_model() first.")

        self.full_model = self._prepare_full_model(
            base_model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)






    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    