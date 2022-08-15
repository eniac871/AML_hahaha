import imp
import os
import numpy as np
import xgboost as xgb
from PIL import Image
import pandas as pd
import mlflow
#from azureml.core import Model


def init():
    global xgb_model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder (./azureml-models)
    # Please provide your model's folder name if there's one
    # model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.json")

    model_path = "C:/code/AML_hahaha/endpoint/batch/custom_model/model/modelforDummies.model"
    #mlflow.log_text( os.listdir(os.environ["AZUREML_MODEL_DIR"]), "custome_log.txt")
    xgb_model = xgb.Booster(model_file='C:/code/AML_hahaha/endpoint/batch/custom_model/model/modelforDummies.model')
    # xgb_model.load_model(model_path)

   
def run(mini_batch):

    #mlflow.log_text(mini_batch, "minibatch.txt")
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []

    for line in mini_batch:
        #mlflow.log_text(line, "minibatch.txt")
        test = pd.read_csv(line)

        # data cleanup
        test.drop("Id", axis=1, inplace=True)
        data = test.reset_index(drop=True)
        #data.drop(['SalePrice'], axis=1, inplace = True)

        # features that can use the most common data or things means average to fillna
        data['Functional'] = data['Functional'].fillna('Typ')
        data['Electrical'] = data['Electrical'].fillna('SBrkr')
        data['KitchenQual'] = data['KitchenQual'].fillna('TA')
        data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
        data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

        # show then remove house without pool but has poorArea
        pd.options.display.max_columns=None
        data[data['PoolArea'] > 0 & data['PoolQC'].isnull()][["PoolArea", "PoolQC"]]

        # data.loc[2418, 'PoolQC'] = 'Fa'
        # data.loc[2501, 'PoolQC'] = 'Gd'
        # data.loc[2597, 'PoolQC'] = 'Fa'
        data[(data['GarageYrBlt']>2022)]

        location = data[data['GarageYrBlt']>2022]
        location
        data.loc[data['GarageYrBlt']>2022,'GarageYrBlt'] = 2007

        pd.options.display.max_columns=None
        pd.options.display.max_rows=None
        data[data['GarageYrBlt'].notnull() & data['GarageType'].isnull()][['GarageYrBlt', "GarageType"]]

        temp = data[data['MSZoning'].isnull()]
        print("NA counts for MSZoning is {}", format(temp.shape[0]))
        # temporarily use mode to fill MSZoning NA accordiing to mssubclass
        msclass_group = data.groupby('MSSubClass')
        zoning = msclass_group['MSZoning'].apply(lambda x :x.mode()[0])
        data['MSZonging'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        i=0
        #while i < data.shape[1]:
            #data.iloc[:,i].fillna(data.iloc[:,i].mode()[0])
            #i +=1
        while i < data.shape[1]:
            data.loc[:, data.columns[i]] = data[data.columns[i]].fillna(data.iloc[:,i].mode()[0])
            i = i+1
        test_dummies = pd.get_dummies(data).reset_index(drop=True)



        xgbdata = xgb.DMatrix(test_dummies)
        test_predict = xgb_model.predict(xgbdata)
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

def predict_data():
    testdata = pd.read_csv("C:/code/AML_hahaha/endpoint/batch/endpoint_data/test_dummies.csv")
    run(["C:/code/AML_hahaha/data/house-prices-advanced-regression-techniques/test.csv"])


init()

predict_data()
