# This is largely based on GEMBA-MQM: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba_mqm.py

import pandas as pd
from llm_fewshots_examples import *
from openai import OpenAI
from collections import defaultdict
import argparse
import os

lang_dict={
    "en": "English",
    "de": "German"
}

def apply_template(template, data):
    if isinstance(template, str):
        return template.format(**data)
    elif isinstance(template, list):
        prompt = []
        for conversation_turn in template:
            p = conversation_turn.copy()
            p['content'] = p['content'].format(**data)
            prompt.append(p)
        return prompt
    else:
        raise ValueError(f"Unknown template type {type(template)}")

def get_bilingual_context(df, doc_id, seg_id, k):
  context_text = []
  for con_seg_id in range(max(0, seg_id-k), seg_id):
    row = df[(df["doc_id"]==doc_id) & (df["segment_id"]==con_seg_id) & (df["model"]=="baseline")].values
    assert len(row) == 1
    context_text.append(f"{row[0][2]} ({row[0][-2]}): {row[0][6]}")
  return ("\n").join(context_text)

def get_response(client, prompt):
  parameters = {
              "temperature": 0,
              "max_tokens": 100,
              "top_p": 1,
              "n": 1,
              "frequency_penalty": 0,
              "presence_penalty": 0,
              "stop": None,
              "model": "gpt-4",
              "messages": prompt,
          }
  response = client.chat.completions.create(**parameters)
  return response.choices[0].message.content.strip()

def parse_error_class(error):
    # parse error from error description, errors are ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']
    #  locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use),
    class_name = "unknown"
    if "accuracy" in error:
        class_name = "accuracy"
        for subclass in ["addition", "mistranslation", "omission", "untranslated text"]:
            if subclass in error:
                class_name = f"accuracy-{subclass}"
    elif "fluency" in error:
        class_name = "fluency"
        for subclass in ["character encoding", "grammar", "inconsistency", "punctuation", "register", "spelling"]:
            if subclass in error:
                class_name = f"fluency-{subclass}"
    elif "locale convention" in error:
        class_name = "locale convention"
        for subclass in ["currency", "date", "name", "telephone", "time"]:
            if subclass in error:
                class_name = f"locale convention-{subclass}"
    elif "style" in error:
        class_name = "style"
    elif "terminology" in error:
        class_name = "terminology"
        for subclass in ["inappropriate", "inconsistent"]:
            if subclass in error:
                class_name = f"terminology-{subclass}"
    elif "non-translation" in error:
        class_name = "non-translation"
    elif "other" in error:
        class_name = "other"

    return class_name

def parse_mqm_answer(x, full_desc=True):
    if x is None:
        return None

    x = str(x)
    if x.startswith('{"improved translation"'):
      print("here")
    else:
        x = x.lower()
        errors = {'critical': [], 'major': [], 'minor': []}
        error_level = None
        for line in x.split('\n'):
            line = line.strip()
            if "no-error" in line or "no error" in line or "no errors" in line or "" == line:
                continue
            if "critical:" == line:
                error_level = "critical"
                continue
            elif "major:" == line:
                error_level = "major"
                continue
            elif "minor:" == line:
                error_level = "minor"
                continue

            if "critical" in line or "major" in line or "minor" in line:
                if not any([line.startswith(x) for x in ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']]):
                    print(line)

            if error_level is None:
                print(f"No error level for {line}")
                continue

            if "non-translation" in line:
                errors["critical"].append(line)
            else:
                errors[error_level].append(line)

    error_classes = defaultdict(list)
    final_score = 0
    error_counter = {'critical':0, 'major':0, 'minor':0}
    for error_level in ['critical', 'major', 'minor']:
        if error_level not in errors:
                continue
        for error in errors[error_level]:
            final_score += 10 if error_level == 'critical' else 5 if error_level == 'major' else 1
            error_counter[error_level] += 1

            if full_desc:
                error_classes[error_level].append(error)
            else:
                class_name = parse_error_class(error)
                error_classes[error_level].append(class_name)

    # We remove this for chat data as human annotations were collected without this constraint unlike other WMT tasks
    # if final_score > 25:
    #     final_score = 25

    return pd.Series([-final_score, error_counter['critical'], error_counter['major'], error_counter['minor']])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/all_data.csv")
    parser.add_argument("--hypothesis_file", type=str, default=None)
    parser.add_argument("--model_name", type=str,  default="gpt-4")
    parser.add_argument("--lp", default="en_de", type=str)
    parser.add_argument("--out_file", default=None, type=str)
    parser.add_argument('--compute_corr', action='store_true')
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument("--N", type=int, default=1000)
    args = parser.parse_args()
    return args

def main(args):
    credentials = {
        "deployments": {args.model_name: args.model_name},
        "api_key": os.getenv('OPENAI_API_KEY'),
        "requests_per_second_limit": 1
    }

    client = OpenAI(api_key=credentials["api_key"],
                    organization=credentials['organization'],)

    dfs_all = pd.read_csv(args.input_csv, index_col=None)
    dfs_all["src_len"] = dfs_all["source"].apply(lambda x: len(x.split(" ")))

    df_subset = dfs_all[(dfs_all.lp==args.lp) & (dfs_all["src_len"] > 1 )].sample(args.N)
    df_subset["sender"] = df_subset["sender"].replace("agent", "Agent")
    df_subset["sender"] = df_subset["sender"].replace("client", "Customer")

    df_subset.rename(columns={'source': 'source_seg',
                            'mt': 'target_seg',}, inplace=True)

    df_subset['source_lang'] = lang_dict[args.lp.split("_")[0]]
    df_subset['target_lang'] = lang_dict[args.lp.split("_")[0]]

    context = []
    for _, row in df_subset.iterrows():
        context.append(get_bilingual_context(dfs_all, row["doc_id"], row["segment_id"], args.context_size))
    df_subset["context"] = context

    df_subset["prompt-3shot"] = df_subset.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM_1shot, x), axis=1)
    df_subset["context_prompt-3shot"] = df_subset.apply(lambda x: apply_template(TEMPLATE_GEMBA_CONTEXT_MQM_1shot, x), axis=1)

    df_subset["GPT-4-annotations-1shot"] = df_subset["prompt-1shot"].apply(lambda x: get_response(client, x))
    df_subset["GPT-4-context-annotations-1shot"] = df_subset["context_prompt"].apply(lambda x: get_response(client, x))

    df_subset[["GPT-4-context-1shot-score", "GPT-4-context-1shot-critical-count", "GPT-4-context-1shot-major-count", "GPT-4-context-1shot-minor-count"]] = df_subset["GPT-4-context-annotations-1shot"].apply(parse_mqm_answer)
    df_subset[["GPT-4-1shot-score", "GPT-4-1shot-critical-count", "GPT-4-1shot-major-count", "GPT-4-1shot-minor-count"]] = df_subset["GPT-4-annotations-1shot"].apply(parse_mqm_answer)

    df_subset.to_csv(args.out_file, index=None)


if __name__ == "__main__":
    args = get_args()
    main(args)