from Sign_Language.constants import *
from Sign_Language.utils.common import read_yaml, create_directories
from Sign_Language.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            raw_data_path=config.raw_data_path 
        )

        return data_ingestion_config
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        params = self.params.baseModelParams


        get_prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
           
            NumOfClasses = params.NumOfClasses,
            input_shape = params.input_shape,
            Conv1D_1 = params.Conv1D_1,
            Dropout1 = params.Dropout1,
            Conv1D_2 = params.Conv1D_2,
            Dropout2 = params.Dropout2,
            pool_size = params.pool_size,
            Dense_layer = params.Dense_layer
        )

        return get_prepare_base_model_config