from Sign_Language.config.configuration import ConfigurationManager
from Sign_Language.components.prepare_base_model import PrepareModel
from Sign_Language import logger


STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        get_prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareModel(config=get_prepare_base_model_config)
        prepare_base_model.create_model()
        


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e