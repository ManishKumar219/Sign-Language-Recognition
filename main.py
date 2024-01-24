from Sign_Language import logger
from Sign_Language.pipeline.stage01_DataIngestion import DataIngestionTrainingPipeline
from Sign_Language.pipeline.stage02_Prepare_Base_Model import PrepareBaseModelPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Prepare Base Model"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   obj = PrepareBaseModelPipeline()
   obj.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e