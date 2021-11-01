import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml,create_directory

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
STAGE="two"
def main(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    prepare_data_dir=os.path.join(artifacts_dir,  artifacts["PREPARED_DATA"])
    train_data_path= os.path.join(prepare_data_dir,artifacts["TRAIN_DATA"])
    test_data_path= os.path.join(prepare_data_dir,artifacts["TEST_DATA"])
    logging.info(train_data_path)
    print(train_data_path)
    logging.info(test_data_path)

    feature_data_dir=os.path.join(artifacts_dir,  artifacts["FEATURED_DATA_DIR"])
    create_directory([feature_data_dir])
    feature_train=os.path.join(feature_data_dir,artifacts["FEATURE_OUT_TRAIN"])
    feature_test=os.path.join(feature_data_dir,artifacts["FEATURE_OUT_TEST"])
    

    create_directory()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e