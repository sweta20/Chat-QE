import argparse
from comet import download_model, load_from_checkpoint
from typing import List
import pandas as pd
from scipy import stats
import random

def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output

# Referred from https://github.com/amazon-science/doc-mt-metrics/blob/main/Prism/add_context.py
def add_context(orig_txt: List[str], context_same: List[str], context_other: List[str], 
                sender_ids: List[str], sep_token: str = "</s>", ws: int = 2, 
                add_noise=False, noise_type=None, drop_indices=None, context_type="across") -> List[str]:
    if not (len(orig_txt) == len(context_same)== len(context_other)):
        raise Exception(f'Lengths should match: len(orig_txt)={len(orig_txt)}, len(context_same)={len(context_same)}, len(context_other)={len(context_other)}')
    context_all = []
    for i in range(len(orig_txt)):
      context_window = []
      for j in range(max(0, i - ws), i):
        if context_type=="across":
          if sender_ids[j] == sender_ids[i]:
            context_window.append(context_same[j])
          else:
              context_window.append(context_other[j])
        else:
          context_window.append(context_same[j])
      context_all.append(context_window)
      
    if add_noise and noise_type=="shuffle":
      random.shuffle(context_all)

    augm_txt = []
    for i in range(len(orig_txt)):
      context = context_all[i]
      if noise_type=="drop" and drop_indices[i] <  len(context):
        if len(context) > 0:
          context.pop(drop_indices[i])
        else:
          print("No context to drop")
      augm_txt.append(" {} ".format(sep_token).join(context + [orig_txt[i]]))

    return augm_txt

class DocCometMetric():
  def __init__(self, model_name="Unbabel/wmt20-comet-qe-da", batch_size=64, ref_based=False, enable_context=True):
    checkpoint_path = download_model(model_name)
    self.model = load_from_checkpoint(checkpoint_path)
    self.batch_size = batch_size
    if enable_context:
      print("Context Enabled")
      self.model.enable_context()
    self.ref_based = ref_based

  def get_score(self, source, outputs, references):
    if not self.ref_based:
      del references
      print("Using QE")
      return self.model.predict([{"mt": y, "src": x} for x, y in zip(source, outputs)],
        batch_size=self.batch_size, gpus=1, progress_bar=True)['scores']
    else:
       return self.model.predict([{"mt": y, "ref":z, "src": x} for x, y, z in zip(source, outputs, references)],
        batch_size=self.batch_size, gpus=1, progress_bar=False)['scores']
       
  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/all_data.csv")
    parser.add_argument("--hypothesis_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Unbabel/wmt20-comet-qe-da")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--context_type", type=str, default="across")
    parser.add_argument('--ref-based', action='store_true')
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--mt_context", type=str, default="mt")
    parser.add_argument("--sender-col", type=str, default="sender")
    parser.add_argument('--compute_corr', action='store_true')
    parser.add_argument('--disable_context', action='store_true')
    parser.add_argument("--out_file", default=None, type=str)
    parser.add_argument("--add_noise", action='store_true')
    parser.add_argument("--noise_type", default=None, type=str)

    args = parser.parse_args()
    return args

def main(args):

    df = pd.read_csv(args.input_csv)

    if args.hypothesis_file is not None:
      df["mt"] = read_file(args.hypothesis_file)

    comet_metric = DocCometMetric(model_name=args.model_name, 
                                  batch_size=args.batch_size, 
                                  ref_based=args.ref_based,
                                  enable_context=not args.disable_context)

    groupby_cols = ["doc_id", "model"]
      
    doc_dfs = []
    for _, df_group in df.groupby(groupby_cols):
        df_group = df_group.sort_values(['segment_id'])
        if args.add_noise and args.noise_type=="drop_pair":
          drop_indices  = [random.randint(0, args.context_size-1) for i in range(len(df_group["source"]))]
        else:
          drop_indices = None

        df_group[f"source_with_context"]  = add_context(
                                                orig_txt=df_group["source"].to_list(), 
                                                context_same=df_group["source"].to_list(), 
                                                context_other=df_group[args.mt_context].to_list(), 
                                                sender_ids=df_group[args.sender_col].to_list(), 
                                                sep_token=comet_metric.model.encoder.tokenizer.sep_token, 
                                                ws=args.context_size,
                                                add_noise=args.add_noise,
                                                noise_type=args.noise_type,
                                                drop_indices=drop_indices, 
                                                context_type=args.context_type)
        df_group[f"mt_with_context"]  = add_context(
                                                orig_txt=df_group["mt"].to_list(), 
                                                context_same=df_group[args.mt_context].to_list(), 
                                                context_other=df_group["source"].to_list(), 
                                                sender_ids=df_group[args.sender_col].to_list(), 
                                                sep_token=comet_metric.model.encoder.tokenizer.sep_token, 
                                                ws=args.context_size,
                                                add_noise=args.add_noise,
                                                noise_type=args.noise_type,
                                                drop_indices=drop_indices,
                                                context_type=args.context_type)
        df_group[f"ref_with_context"]  = add_context(
                                                orig_txt=df_group["reference"].to_list(), 
                                                context_same=df_group["reference"].to_list(), 
                                                context_other=df_group["source"].to_list(), 
                                                sender_ids=df_group[args.sender_col].to_list(), 
                                                sep_token=comet_metric.model.encoder.tokenizer.sep_token, 
                                                ws=args.context_size,
                                                add_noise=args.add_noise,
                                                noise_type=args.noise_type,
                                                drop_indices=drop_indices,
                                                context_type=args.context_type)
        doc_dfs.append(df_group)

    dfs_all = pd.concat(doc_dfs)
    dfs_all["metric"] = comet_metric.get_score(dfs_all[f'source_with_context'].to_list(), dfs_all[f'mt_with_context'].to_list(), dfs_all[f'ref_with_context'].to_list())

    if args.out_file is not None:
        dfs_all.to_csv(args.out_file, index=None)

    if args.compute_corr:
      for lp_name, lp_df in dfs_all.groupby("lp"):
        print(lp_name, args.context_size, args.context_type, stats.spearmanr(lp_df["metric"], lp_df["google_mqm"]).statistic)
       

if __name__ == "__main__":
    args = get_args()
    main(args)