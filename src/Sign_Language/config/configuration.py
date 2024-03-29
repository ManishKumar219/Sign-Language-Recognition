from Sign_Language.constants import *
from Sign_Language.utils.common import read_yaml, create_directories
from Sign_Language.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, 
                                                PrepareCallbacksConfig, TrainingConfig, 
                                                EvaluationConfig)
import os

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
    
    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:

        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])


        get_prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )

        return get_prepare_callbacks_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params.baseModelParams

        training_data = Path(self.config.data_ingestion.train_data_path)

        create_directories([
            Path(training.root_dir)
        ])
        
        training_config = TrainingConfig(
            root_dir = Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            base_model_path = Path(prepare_base_model.base_model_path),
            training_data = training_data,
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            )
        
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:

        config = self.config
        # path_of_model = self.

        eval_config = EvaluationConfig(
            path_of_model = Path(config.training.trained_model_path),
            testing_data = r"artifacts\data-ingestion\test.csv",
            all_params = self.params.baseModelParams,
            params_batch_size = self.params.baseModelParams.BATCH_SIZE
        )

        return eval_config