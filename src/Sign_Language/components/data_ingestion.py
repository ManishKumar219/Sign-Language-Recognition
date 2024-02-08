from Sign_Language.entity.config_entity import (DataIngestionConfig)
import os
import pandas as pd
from Sign_Language import logger
from Sign_Language.utils.common import get_size
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def load_file(self):
        if not os.path.exists(self.config.raw_data_path):
            df = pd.read_csv("research/data/Data.csv", index_col = "Unnamed: 0")
            logger.info(f"Read the Raw dataset as DataFrame: {self.config.raw_data_path}")

            df.to_csv(self.config.raw_data_path, index=True, header=True)

            logger.info(f"loaded the raw dataset: {self.config.raw_data_path}")

        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.raw_data_path))}")

    def create_train_test_data(self):
        if not os.path.exists(self.config.train_data_path):
            df = pd.read_csv(self.config.raw_data_path, index_col = "Unnamed: 0")
            traindf, testdf = train_test_split(df, test_size=0.2, random_state=12)

            traindf.to_csv(self.config.train_data_path, index=True, header=True)
            testdf.to_csv(self.config.test_data_path, index=True, header=True)

            logger.info("Created the training and test data")
        else:
            logger.info(f"File already exists of size: \n \
                        {get_size(Path(self.config.train_data_path))}\n\
                        {get_size(Path(self.config.train_data_path))}")