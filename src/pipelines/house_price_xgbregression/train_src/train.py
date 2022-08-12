import argparse
import numpy as np
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import mlflow

mlflow.sklearn.autolog()

parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--price_data", type=str, help="Path to price data")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

print("hello training world...")


print("mounted_path files: ")


df_list = []
arr = os.listdir(args.training_data)
print(arr)
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.training_data, filename), "r") as handle:
        # print (handle.read())
        input_df = pd.read_csv((Path(args.training_data) / filename))
        df_list.append(input_df)

arr = os.listdir(args.price_data)
print(arr)
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.price_data, filename), "r") as handle:
        # print (handle.read())
        input_df = pd.read_csv((Path(args.price_data) / filename))
        df_list.append(input_df)

train_dummies = df_list[0]
y = df_list[1]
y["SalePrice"] = np.log1p(y["SalePrice"])

print(train_dummies.shape)
print(y.shape)
xgbmodel = xgb.XGBRegressor(max_depth=5, n_estimators = 400)
model = xgbmodel.fit(train_dummies, y)
mean = np.sqrt(-cross_val_score(xgbmodel, train_dummies, y, cv=5, scoring = "neg_mean_squared_error")).mean()

print("np is: " + str(mean))
print("score " + str(model.score(train_dummies, y)))

mlflow.sklearn.save_model(model, args.model_output)

