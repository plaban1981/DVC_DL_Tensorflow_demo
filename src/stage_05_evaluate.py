from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
from sklearn.metrics import roc_curve,auc,accuracy_score
import argparse
import os
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    base_model_dir = artifacts['TRAINED_MODEL_DIR']
    updated_model = artifacts["TRAINED_MODEL"]


    train_model_dir_path = os.path.join(artifacts_dir,base_model_dir)
    print(train_model_dir_path)
    model_pickle = os.listdir(train_model_dir_path)[0]
    trained_model_path = os.path.join(artifacts_dir,base_model_dir,model_pickle)
    model = load_full_model(trained_model_path)


    #untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    #model = load_full_model(untrained_full_model_path)
    logging.info(f">>>>untrained model loading complete")
    #model.load_weights(train_model_dir_path)
    logging.info(f">>>>load trained model weights")

    #callback_dir_path  = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    #callbacks = get_callbacks(callback_dir_path)

    _, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )
    
    #steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    preds = model.predict(valid_generator,verbose=1)
    predictions = np.argmax(preds,axis=1)
    logging.info(f">>>>> Evaluation of the trained model completed")
    print(predictions)
    fpr, tpr, _ = roc_curve(valid_generator.classes, predictions )
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(valid_generator.classes, predictions)
    logging.info(f"\n\n>>>>> roc auc value of validation data : {roc_auc}")
    logging.info(f"\n\n>>>>> validation data accuracy  : {accuracy}")



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage five started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage five completed! Model Evaluation Complete >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e