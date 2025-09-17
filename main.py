from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline

def get_args():
    parser = argparse.ArgumentParser(description="Choose model for training")
    parser.add_argument("--model", type=str, required=True, help="Model to use: EfficientNetB0 or ResNet50")
    return parser.parse_args()

def main():
    args = get_args()
    model_name = args.model
    print(f"🔹 Selected model: {model_name}")

    if model_name=="EffecientNetB0":
         STAGE_NAME = "Prepare base model"
         try:
            logger.info(f"*******************")
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            prepare_base_model = PrepareBaseModelTrainingPipeline()
            prepare_base_model.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
         except Exception as e:
                 logger.exception(e)
                 raise e
         STAGE_NAME = "Training"
         try:
            logger.info(f"*******************")
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            model_trainer = ModelTrainingPipeline()
            model_trainer.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
         except Exception as e:
                 logger.exception(e)
                 raise e
         
      if model_name=="ResNet50":
         STAGE_NAME = "Prepare base model"
         try:
            logger.info(f"*******************")
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            prepare_base_model = PrepareBaseModelResNetTrainingPipeline()
            prepare_base_model.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
         except Exception as e:
                 logger.exception(e)
                 raise e  
         STAGE_NAME = "Training"
         try:
            logger.info(f"*******************")
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            model_trainer = ModelTrainingResNetPipeline()
            model_trainer.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
         except Exception as e:
                 logger.exception(e)
                 raise e
         



# STAGE_NAME = "Prepare base model"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_base_model = PrepareBaseModelTrainingPipeline()
#    prepare_base_model.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


# STAGE_NAME = "Training"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    model_trainer = ModelTrainingPipeline()
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

if __name__ == "__main__":
    main()