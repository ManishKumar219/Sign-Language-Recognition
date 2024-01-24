# #Component
from Sign_Language import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from Sign_Language.config.configuration import PrepareBaseModelConfig
from pathlib import Path


class PrepareModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def create_model(self):
        model = Sequential()

        # Convolutional Layer 1
        model.add(Conv1D(filters=self.config.Conv1D_1, kernel_size=3, activation='relu', input_shape=(self.config.input_shape, 1)))
        model.add(Dropout(self.config.Dropout1))

        # Convolutional Layer 2
        model.add(Conv1D(filters=self.config.Conv1D_2, kernel_size=3, activation='relu'))
        model.add(Dropout(self.config.Dropout2))

        # Max Pooling Layer
        model.add(MaxPooling1D(pool_size=2))

        # Flatten Layer
        model.add(Flatten())

        # Fully Connected Layer 1
        model.add(Dense(self.config.Dense_layer, activation='relu'))

        # Fully Connected Layer 2
        model.add(Dense(28, activation='softmax'))

        # Print model summary
        model.summary()
        self.save_model(self.config.base_model_path, model)
        logger.info("Created the Base Model:")

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

