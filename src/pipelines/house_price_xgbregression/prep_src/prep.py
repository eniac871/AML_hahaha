import argparse
from multiprocessing.spawn import prepare
from pathlib import Path
from typing_extensions import Concatenate
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser("prep")
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")
parser.add_argument("--price_data", type=str, help="Path of price_data")

args = parser.parse_args()

print("hello training world...")

lines = [f"Raw data path: {args.raw_data}", f"Data output path: {args.prep_data}"]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.raw_data)
print(arr)

df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.raw_data, filename), "r") as handle:
        # print (handle.read())
        # ('input_df_%s' % filename) = pd.read_csv((Path(args.training_data) / filename))
        input_df = pd.read_csv((Path(args.raw_data) / filename))
        df_list.append(input_df)


sample_submission = df_list[0]
test = df_list[1]
train = df_list[2]

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

print("origin train data size is {}", format(train.shape))
print("origin test data size is {}", format(test.shape))
trainId = train['Id']
testId = test['Id']
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

print("train data size after drop id is {}", format(train.shape))
print("test data size after drop id is {}", format(test.shape))

caseToDrop = train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)]
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#  delete 2 case
data = pd.concat((train, test)).reset_index(drop=True)
print("combine data size is {}", format(data.shape))
data.drop(['SalePrice'], axis=1, inplace = True)
print("combine data without SalePrice, size is {}", format(data.shape))
#combine train and test in new data


# features that can use the most common data or things means average to fillna
data['Functional'] = data['Functional'].fillna('Typ')
data['Electrical'] = data['Electrical'].fillna('SBrkr')
data['KitchenQual'] = data['KitchenQual'].fillna('TA')
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

# show then remove house without pool but has poorArea
pd.options.display.max_columns=None
data[data['PoolArea'] > 0 & data['PoolQC'].isnull()][["PoolArea", "PoolQC"]]

data.loc[2418, 'PoolQC'] = 'Fa'
data.loc[2501, 'PoolQC'] = 'Gd'
data.loc[2597, 'PoolQC'] = 'Fa'
data[(data['GarageYrBlt']>2022)]
print("data before drop error year size is {}", format(data.shape))
location = data[data['GarageYrBlt']>2022]
location
data.loc[data['GarageYrBlt']>2022,'GarageYrBlt'] = 2007
# data = data.drop(data[(data['GarageYrBlt']>2022)].index)
# print("data after drop error year size is {}", format(data.shape))

pd.options.display.max_columns=None
pd.options.display.max_rows=None
data[data['GarageYrBlt'].notnull() & data['GarageType'].isnull()][['GarageYrBlt', "GarageType"]]
# No one without garage has garageyrBlt, nothing to do
# consider whether we need to check garage condition later.()

temp = data[data['MSZoning'].isnull()]
print("NA counts for MSZoning is {}", format(temp.shape[0]))
# temporarily use mode to fill MSZoning NA accordiing to mssubclass
msclass_group = data.groupby('MSSubClass')
zoning = msclass_group['MSZoning'].apply(lambda x :x.mode()[0])
data['MSZonging'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# data.iloc[:,0]
i=0
#while i < data.shape[1]:
    #data.iloc[:,i].fillna(data.iloc[:,i].mode()[0])
    #i +=1
while i < data.shape[1]:
    data.loc[:, data.columns[i]] = data[data.columns[i]].fillna(data.iloc[:,i].mode()[0])
    i = i+1

temp = data.isnull().sum()
ans = temp.drop(temp[temp == 0].index).sort_values(ascending = False)
len(ans)
# fill all na now if len = 0
data_dummies = pd.get_dummies(data).reset_index(drop=True)
data_dummies.shape

train_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice"], axis=1), test), axis=0)).iloc[: train.shape[0]]
test_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice"], axis=1), test), axis=0)).iloc[train.shape[0]:]
# train_dummies = data_dummies.iloc[: train.shape[0]]
# test_dummies = data_dummies.iloc[train.shape[0]]

print(train_dummies.shape)

# log price
# price_data = np.log1p(train["SalePrice"])
price_data = train['SalePrice']

print(price_data.shape)
output_price = train_dummies.to_csv((Path(args.prep_data) / "house_price_prep_data.csv"))
sale_price  = price_data.to_csv((Path(args.price_data) / "house_price_price_data.csv"))

