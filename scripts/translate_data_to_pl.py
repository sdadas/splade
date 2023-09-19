import json
import logging
import os.path
import re
import shutil
import sys
from pathlib import Path
from typing import Dict

from datasets import load_dataset


class DataTranslator:

    def __init__(self, src_dir: str, dst_dir: str, msmarco_dir: str):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.queries = self._load_msmarco_data("queries")
        self._load_additional_queries(self.queries, msmarco_dir)
        self.passages = self._load_msmarco_data("corpus")

    def _load_msmarco_data(self, part: str):
        logging.info(f"Loading MS Marco {part}")
        rows = load_dataset("clarin-knext/msmarco-pl", name=part, split=part)
        results = {}
        for batch in rows.iter(batch_size=1000):
            for idx in range(len(batch["_id"])):
                text = batch["text"][idx].replace("\n", " ").replace("\r", " ").replace("\t", " ")
                pid = int(batch["_id"][idx])
                results[pid] = text
        return results

    def _load_additional_queries(self, queries: Dict, msmarco_dir: str):
        for file_name in ("queries.dev.jsonl", "queries.eval.jsonl", "queries.train.jsonl"):
            file_path = os.path.join(msmarco_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    value = json.loads(line.strip())
                    rowid = int(value["id"])
                    if rowid not in queries:
                        queries[rowid] = re.sub(r"[\n\r\t]+", " ", value["translation"]).lower()
    def copy_dir(self, src_name: str):
        src_path = os.path.join(self.src_dir, src_name)
        dst_path = os.path.join(self.dst_dir, src_name)
        logging.info(f"Copying dir {src_path} to {dst_path}")
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    def copy_translate(self, src_name: str):
        src_path = os.path.join(self.src_dir, src_name)
        for root, dirs, files in os.walk(src_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.src_dir)
                dst_path = os.path.join(self.dst_dir, rel_path)
                parent = Path(dst_path).parent
                parent.mkdir(parents=True, exist_ok=True)
                if file.endswith(".tsv"):
                    if "collection" in parent.name:
                        self._translate_file(file_path, dst_path, self.passages)
                    else:
                        self._translate_file(file_path, dst_path, self.queries)
                else:
                    shutil.copy(file_path, dst_path)

    def _translate_file(self, src_path: str, dst_path: str, collection: Dict):
        logging.info(f"Translating file {src_path} to {dst_path}")
        with open(src_path, "r", encoding="utf-8") as infile, open(dst_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                values = [val.strip() for val in line.strip().split("\t", maxsplit=1)]
                rowid = int(values[0])
                text = collection[rowid]
                outfile.write(f"{rowid}\t{text}\n")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(logging.INFO)
    input_dir, output_dir, msmarco_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    translator = DataTranslator(input_dir, output_dir, msmarco_dir)
    translator.copy_translate("all_dev_queries")
    translator.copy_translate("dev_queries")
    translator.copy_translate("train_queries")
    translator.copy_translate("val_retrieval")
    translator.copy_translate("full_collection")
    translator.copy_dir("hard_negatives_scores")
    translator.copy_dir("triplets")
    translator.copy_dir("vienna_triplets")
    shutil.copy(os.path.join(input_dir, "dev_qrel.json"), os.path.join(output_dir, "dev_qrel.json"))
