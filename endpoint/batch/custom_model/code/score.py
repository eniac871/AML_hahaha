import imp
import os
import numpy as np
import xgboost as xgb
from PIL import Image
import pandas as pd
from azureml.core import Model


def init():
    global xgb_model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder (./azureml-models)
    # Please provide your model's folder name if there's one
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.json")

    # model_path = "../model/model.json"
    print(os.listdir(os.environ["AZUREML_MODEL_DIR"]))
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)

   
def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []

    for line in mini_batch:
        data_to_test = pd.read_csv(line)
        test_predict = xgb_model.predict(data_to_test)
        resultList += test_predict.tolist()
    return resultList
    

    # for line in mini_batch:
    #     test_predict = xgb_model.predict(line)
    #     # prepare each image
    #     data = Image.open(image)
    #     np_im = np.array(data).reshape((1, 784))
    #     # perform inference
    #     inference_result = output.eval(feed_dict={in_tensor: np_im}, session=g_tf_sess)
    #     # find best probability, and add to result list
    #     best_result = np.argmax(inference_result)
    #     resultList.append("{}: {}".format(os.path.basename(image), best_result))

    # return resultList

# def predict_data():
#     testdata = pd.read_csv("../../endpoint_data/test_dummies.csv")
#     run(testdata)

# init()

# predict_data()