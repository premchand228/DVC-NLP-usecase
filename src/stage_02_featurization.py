import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
import logging
from src.utils.common import read_yaml,create_directory,get_df
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
    max_features=params["featurize"]["max_features"]
    ngrams=params["featurize"]["ngrams"]

    df_train=get_df(train_data_path)

    train_words = np.array(df_train.text.str.lower().values.astype("U"))
    print(train_words[0:20])

    #create_directory() 
    bag_of_words=CountVectorizer(stop_words="english",max_features=max_features,ngrams_range=(1,ngrams))
    bag_of_words.fit(train_words)
    train_words_binary_matrix=bag_of_words.transform(train_words)

    tfidf=TfidfVectorizer(smooth_idf=False)

    ## Smooth idf weights by adding one to document frequencies, 
    # as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.

    tfidf.fit(train_words_binary_matrix)
    train_words_tf_idf_matrix=tfidf.transform(train_words_binary_matrix)
    

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