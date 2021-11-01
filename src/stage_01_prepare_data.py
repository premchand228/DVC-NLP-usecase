import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml,create_directory
from src.utils.data_managemnt import process_posts
import random



logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
STAGE="one"    

def main(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    source_data=config["source_data"]
    source_data_dir=source_data["data_dir"]
    source_data_file=source_data["data_file"]
    input_data=os.path.join(source_data_dir,source_data_file)
    split=params["prepare"]["split"]
    seed=params["prepare"]["seed"]
    random.seed(seed)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    prepare_data_dir=os.path.join(artifacts_dir,  artifacts["PREPARED_DATA"])

    create_directory([prepare_data_dir])

    train_data_path= os.path.join(prepare_data_dir,artifacts["TRAIN_DATA"])
    test_data_path= os.path.join(prepare_data_dir,artifacts["TEST_DATA"])
    logging.info(train_data_path)
    print(train_data_path)
    logging.info(test_data_path)
    encode="utf8"
    with open(input_data,encoding=encode) as fd_in:
        with open(train_data_path,"w",encoding=encode) as fd_out_train:
            with open(test_data_path,"w",encoding=encode) as fd_out_test:
                
                process_posts(fd_in,fd_out_train,fd_out_test,"<python>",split)








    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e