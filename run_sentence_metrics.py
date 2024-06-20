import argparse
from metrics import *
import pandas as pd
from scipy import stats
import sys
import numpy as np

def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output
    
  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/all_data.csv")
    parser.add_argument("--hypothesis_file", type=str, default=None)
    parser.add_argument("--model_name", type=str,  default=None)
    parser.add_argument("--metric_name", type=str, default="CometMT")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lp", default="en_de", type=str)
    parser.add_argument("--out_file", default=None, type=str)
    parser.add_argument('--compute_corr', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    df = pd.read_csv(args.input_csv)
    
    if args.hypothesis_file is not None:
        df["mt"] = read_file(args.hypothesis_file)

    df = df[df.lp == args.lp]

    if args.model_name is not None:
        metric = getattr(sys.modules[__name__], args.metric_name)(model_name=args.model_name,
                                                              batch_size=args.batch_size)
    else:
        metric = getattr(sys.modules[__name__], args.metric_name)(batch_size=args.batch_size)

    df[args.metric_name] = metric.get_score(
                    df['source'].to_list(), 
                    df['mt'].to_list(), 
                    df['reference'].to_list())
    
    print(f"{args.metric_name} Score:", np.mean(df[args.metric_name]))

    if args.out_file is not None:
        df[args.metric_name].to_csv(args.out_file, index=None)

    if args.compute_corr:
        print(stats.spearmanr(df[args.metric_name], df["google_mqm"]).statistic)

if __name__ == "__main__":
    args = get_args()
    main(args)