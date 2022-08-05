import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser("transform")
parser.add_argument("--clean_data", type=str, help="Path to prepped data")
parser.add_argument("--transformed_data", type=str, help="Path of output data")

args = parser.parse_args()


lines = [
    f"Clean data path: {args.clean_data}",
    f"Transformed data output path: {args.transformed_data}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr = os.listdir(args.clean_data)
print(arr)

df_list = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.clean_data, filename), "r") as handle:
        # print (handle.read())
        # ('input_df_%s' % filename) = pd.read_csv((Path(args.training_data) / filename))
        input_df = pd.read_csv((Path(args.clean_data) / filename))
        df_list.append(input_df)


# Transform the data
# combined_df = df_list[1]
price_df = df_list[0]
# These functions filter out coordinates for locations that are outside the city border.
print(price_df.columns)

price_df = price_df.fillna(0)
price_df = price_df.replace(0.0, 0)

normalized_df = price_df.astype({"MasVnrArea": "float64"})
print(normalized_df.columns)

# Output data
transformed_data = normalized_df.to_csv(
    (Path(args.transformed_data) / "transformed_data.csv")
)
