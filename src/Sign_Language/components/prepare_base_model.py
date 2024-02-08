# #Component
from Sign_Language import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from Sign_Language.config.configuration import PrepareBaseModelConfig
from pathlib import Path

# #Component

class PrepareModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def create_model(self):
        self.model = Sequential()

        # ---------------------------------- CNN Model ---------------------------------- #
        # Convolutional Layer 1
        self.model.add(Conv1D(filters=self.config.Conv1D_1, kernel_size=3, activation='relu', input_shape=(self.config.input_shape, 1)))
        self.model.add(Dropout(self.config.Dropout1))

        # Convolutional Layer 2
        self.model.add(Conv1D(filters=self.config.Conv1D_2, kernel_size=3, activation='relu'))
        self.model.add(Dropout(self.config.Dropout2))

        # Max Pooling Layer
        self.model.add(MaxPooling1D(pool_size=2))

        # Flatten Layer
        self.model.add(Flatten())

        # Fully Connected Layer 1
        self.model.add(Dense(self.config.Dense_layer, activation='relu'))

        # Fully Connected Layer 2
        self.model.add(Dense(28, activation='softmax'))
        # ---------------------------------------------------------------------------------- #

        # Print model summary
        self.model.summary()
        # Compile model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.save_model(self.config.base_model_path, self.model)
        logger.info("Created the Base Model:")


    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
