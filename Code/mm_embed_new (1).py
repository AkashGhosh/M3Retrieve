import os
import json
import tqdm
import numpy as np
import torch
import argparse
import difflib
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel

from utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE
from mteb.abstasks.TaskMetadata import TaskMetadata

print("Import Done")

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='nvidia/MM-Embed',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output-dir', default='outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages, only used for Quora as it is a symmetric task')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prefix-type', default='query_or_passage', help='prefix type')
parser.add_argument('--dry-run', action='store_true', help='whether to run the script in dry run mode',default='True')

args = parser.parse_args()
base_name: str = args.model_name_or_path.split('/')[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
os.makedirs(args.output_dir, exist_ok=True)

# === Fuzzy Matching Utilities ===

def find_best_match(target_path, all_files, cutoff=0.7):
    """Find the best approximate match for a file path."""
    matches = difflib.get_close_matches(target_path, all_files, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def collect_all_images(image_root):
    all_images = []
    for root, _, files in os.walk(image_root):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), image_root)
            all_images.append(rel_path)
    return all_images

def validate_image_paths(jsonl_path, image_root, all_images, cutoff=0.7):
    valid_entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                img_path = entry.get('image_path')
                if img_path:
                    matched = find_best_match(img_path, all_images, cutoff=cutoff)
                    if matched:
                        entry['image_path'] = os.path.join(image_root, matched)
                        valid_entries.append(entry)
            except json.JSONDecodeError:
                continue
    return valid_entries

# === Model Wrapper ===

class RetrievalModel(DRESModel):
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, cache_dir='../cache_dir')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir='../cache_dir')
        self.prompt = "Retrieve the relevant paragraph that provides an answer to the given medical query and image."
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self._do_encode(queries, max_length=512, is_query=True)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        return self._do_encode(corpus, max_length=4096, is_query=False)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[dict], max_length: int, is_query: bool) -> np.ndarray:
        encoded_embeds = []
        batch_size = 8 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts = input_texts[start_idx: start_idx + batch_size]
            with torch.cuda.amp.autocast():
                if is_query:
                    embeds = self.encoder.encode(batch_input_texts, is_query=True, instruction=self.prompt, max_length=max_length)['hidden_states']
                else:
                    embeds = self.encoder.encode(batch_input_texts, max_length=max_length)['hidden_states']
                encoded_embeds.append(embeds.cpu().numpy())
        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

# === Main Evaluation ===

def main():
    model = RetrievalModel()

    ROOT = '/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/Relevant_Articles_Ret_MultiCare'
    IMAGE_ROOT = '/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/MultiCare/temp2/images'  # <-- Change as needed

    all_available_images = collect_all_images(IMAGE_ROOT)

    folders = [
        "Cardiology",
        "Dermatology",
        "Endocrinology",
        "Gastroenterology",
        "Genetics and Genomics",
        "Hematology",
        "Infectious Diseases",
        "Neurology",
        "Obstetrics and Gynecology",
        "Oncology",
        "Orthopedics",
        "Pathology",
        "Psychiatry and Behavioral Sciences",
        "Pulmonology",
        "Rheumatology and Immunology"
    ]

    for name in folders:
        queries_file = os.path.join(ROOT, name, 'queries.jsonl')
        valid_entries = validate_image_paths(queries_file, IMAGE_ROOT, all_available_images, cutoff=0.75)

        # Overwrite with filtered and corrected entries
        with open(queries_file, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(json.dumps(entry) + '\n')

        class Custom_Class(AbsTaskRetrieval):
            metadata = TaskMetadata(
                name=f"{name}",
                dataset={
                    "path": os.path.join(ROOT, name),
                    "revision": "None",
                    "model_type": "mm_embed",
                },
                description="Evaluation benchmark on medical image-text retrieval.",
                reference="https://allenai.org/data/scidocs",
                type="Retrieval",
                category="s2p",
                eval_splits=["test"],
                eval_langs=["eng-Latn"],
                main_score="ndcg_at_10"
            )

        evaluation = MTEB(tasks=[Custom_Class()])
        results_dict = evaluation.run(
            model=model,
            output_folder=f'./results_new/MultiCare/mm_embed/{name}',
            eval_splits=["test"],
            overwrite_results=True,
            corpus_chunk_size=400000
        )

        print(results_dict)
        print("==================================================")
        print("Finished evaluation for:", name)

if __name__ == '__main__':
    main()
