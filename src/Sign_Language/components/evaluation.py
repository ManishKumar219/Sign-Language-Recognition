import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from Sign_Language.config.configuration import EvaluationConfig
from Sign_Language.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        data = pd.read_csv(self.config.testing_data, index_col = "Unnamed: 0")
        model = tf.keras.models.load_model(r"artifacts\training\model.h5")
        
        self.X = np.array(data.drop('label', axis = 1))
        self.y = np.array(data[['label']])
        
        self.model = self.load_model(self.config.path_of_model)
        self.score = model.evaluate(self.X, self.y)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)