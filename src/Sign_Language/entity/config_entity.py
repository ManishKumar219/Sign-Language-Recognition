from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    raw_data_path: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    input_shape: int
    NumOfClasses: int
    Conv1D_1: int
    Dropout1: float
    Conv1D_2: int
    Dropout2: float
    pool_size: int
    Dense_layer: int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    testing_data: Path
    all_params: dict
    params_batch_size: int