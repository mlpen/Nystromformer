import os
import argparse
import pandas as pd
import json
from collections import OrderedDict

def read_data(file, end = None):
    with open(file, "r") as f:
        data = f.read().split('\n')
    round_d = {}
    if end is not None:
        for line in data:
            try:
                values = json.loads(line.replace("'",'"'))
                round_d[values["idx"]] = values
            except Exception as e:
                print(e)
    else:
        for line in data:
            try:
                values = json.loads(line.replace("'",'"'))
                round_d[values["idx"]] = values
                if values["idx"] > end:
                    break
            except Exception as e:
                print(e)
    df = pd.DataFrame(round_d).T
    print("Done")
    return df

def get_last_checkpoint(checkpoint_dir):
    try:
        cp_file = sorted([
            file for file in os.listdir(checkpoint_dir)
            if file.startswith("cp-") and file.endswith(".cp")])[-1]
        return os.path.join(checkpoint_dir, cp_file)
    except Exception as e:
        return None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
