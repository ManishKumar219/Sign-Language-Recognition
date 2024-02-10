import tensorflow as tf
import pandas as pd
import numpy as np
from Sign_Language.config.configuration import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_path)
    
    def train_valid_generator(self):

        data = pd.read_csv(self.config.training_data, index_col = "Unnamed: 0")
        self.X = np.array(data.drop('label', axis = 1))
        self.y = np.array(data[['label']])
        
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X, self.y, batch_size=self.config.params_batch_size, 
                       epochs=self.config.params_epochs, validation_split=0.2, 
                       callbacks=callback_list)
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
