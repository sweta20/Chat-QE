from comet import download_model, load_from_checkpoint
import sacrebleu
from bert_score import score as bertscore
from bleurt import score as bleurtscore
import datasets
from metricx23 import models
import transformers
import torch
import pandas as pd

class SentBleurtMT():
    def __init__(self, model_name="BLEURT-20", batch_size=8):
        del batch_size
        self.name = "BLEURTScore"
        self.model = bleurtscore.BleurtScorer(model_name)
        self.model_name = model_name

    def get_score(self, source, output, reference):
        del source
        return self.model.score(candidates=output, references=reference)

class CometQE():
    def __init__(self, model_name="Unbabel/wmt22-cometkiwi-da", batch_size=8):
        self.name = "COMETSrcScore"
        checkpoint_path = download_model(model_name)
        self.model = load_from_checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def get_score(self, source, output, reference=None):
        del reference
        return self.model.predict([{"mt": y, "src": x} for x, y in zip(source, output)],
        batch_size=self.batch_size, gpus=1, progress_bar=False)['scores']

class CometMT():
    def __init__(self, model_name="Unbabel/wmt22-comet-da", batch_size=8):
        self.name = "COMETRefScore"
        checkpoint_path = download_model(model_name)
        self.model = load_from_checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def get_score(self, source, output, reference):
        return self.model.predict([{"mt": y, "ref":z, "src": x} for x, y, z in zip(source, output, reference)],
        batch_size=self.batch_size, gpus=1, progress_bar=False)['scores']


class SentBleu():
    def __init__(self, model_name="", batch_size=8):
        del model_name, batch_size
        self.name = "BLEUScore"

    def get_score(self, source, output, reference):
        del source
        return [sacrebleu.sentence_bleu(x,[y]).score for x, y in zip(output, reference)]


class SentChrf():
    def __init__(self, model_name="", batch_size=8):
        del model_name, batch_size
        self.name = "ChrfScore"

    def get_score(self, source, output, reference):
        del source
        return [sacrebleu.sentence_chrf(x,[y]).score for x, y in zip(output, reference)]


class SentBERTScore():
    def __init__(self, model_name="", batch_size=8):
        del model_name, batch_size
        self.name = "BERTSscore"

    def get_score(self, source, output, reference):
        del source
        _, _, F1 = bertscore(output, reference, lang="en", verbose=True)
        return F1


class MetricXRef():
    def __init__(self, model_name="google/metricx-23-xl-v2p0", batch_size=1):
        self.name = "MetricXRef"
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        self.model = models.MT5ForRegression.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def _make_input(self,example):
        example["input"] = (
          "candidate: "
          + example["hypothesis"]
          + " reference: "
          + example["reference"]
      )
        return example

    def _tokenize(self, example):
        return self.tokenizer(
        example["input"],
        max_length=1024,
        truncation=True,
        padding=False)
    
    def _remove_eos(self, example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def _process_dataset(self, output, reference):
        ds = datasets.Dataset.from_pandas(pd.DataFrame(data=[{"hypothesis": y, "reference": x} for x, y in zip(reference, output)]))
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=0,
            output_all_columns=True,
        )
        return ds

    def get_score(self, source, output, reference):
        del source
        ds = self._process_dataset(output, reference)

        training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=self.batch_size,
            output_dir="./",
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
        )
        predictions, _, _ = trainer.predict(test_dataset=ds)
        return predictions
  
class MetricXQE(MetricXRef):
    def __init__(self,  model_name="google/metricx-23-qe-xl-v2p0", batch_size=1):
        self.name = "MetricXQE"
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        self.model = models.MT5ForRegression.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def _make_input(self,example):
        example["input"] = (
          "candidate: "
          + example["hypothesis"]
          + " source: "
          + example["source"]
      )
        return example

    def _process_dataset(self, source, output):
        ds = datasets.Dataset.from_pandas(pd.DataFrame(data=[{"hypothesis": y, "source": x} for x, y in zip(source, output)]))
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=0,
            output_all_columns=True,
        )
        return ds

    def get_score(self, source, output, reference):
        del reference
        ds = self._process_dataset(source, output)

        training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=self.batch_size,
            output_dir="./",
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
        )
        
        predictions, _, _ = trainer.predict(test_dataset=ds)
        return predictions