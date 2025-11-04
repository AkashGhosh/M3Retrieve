import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel

from utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE
from mteb.abstasks.TaskMetadata import TaskMetadata

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='BAAI/bge-large-en-v1.5',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages, only used for Quora as it is a symmetric task')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prefix-type', default='query_or_passage', help='prefix type')
parser.add_argument('--dry-run', action='store_true', help='whether to run the script in dry run mode', default='True')

args = parser.parse_args()
base_name: str = args.model_name_or_path.split('/')[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
os.makedirs(args.output_dir, exist_ok=True)


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, cache_dir='../cache_dir')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache_dir')
        self.prompt = "Retrieve the relevant paragraph that provides an answer to the given medical query and image:"
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.prompt + q['txt'] for q in queries]
        return self._do_encode(input_texts, max_length=512)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = [str(doc['txt']) for doc in corpus]
        return self._do_encode(input_texts, max_length=512)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], max_length: int) -> np.ndarray:
        encoded_embeds = []
        batch_size = 256 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, max_length=max_length)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = outputs[0][:, 0]
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    model = RetrievalModel()
    folders = ['Pubmed']

    for name in folders:
        class Custom_Class(AbsTaskRetrieval):
            metadata = TaskMetadata(
                name=f"{name}",
                model_type="nv_embed",
                dataset={
                    "path": f"/home/kitsuchart/aakash/MedRAG/clustered_final_dataset/{name}",
                    "revision": "None",
                    "model_type": "bge_large",
                },
                description=(
                    "arguana, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                    " prediction, to document classification and recommendation."
                ),
                reference="",  # external link removed
                type="Retrieval",
                category="s2p",
                eval_splits=["test"],
                eval_langs=["eng-Latn"],
                main_score="ndcg_at_10",
                date=None,
                form=None,
                domains=None,
                task_subtypes=None,
                license=None,
                socioeconomic_status=None,
                annotations_creators=None,
                dialect=None,
                text_creation=None,
                bibtex_citation=None,
                n_samples=None,
                avg_character_length=None,
            )

        evaluation = MTEB(tasks=[Custom_Class()])
        results_dict = evaluation.run(
            model=model,
            output_folder=f'./results./bge_large/{name}',
            eval_splits=["test"],
            overwrite_results=True,
            corpus_chunk_size=400000
        )

        print(results_dict)
        print("==================================================")
        print("Finish evaluation for model:")


if __name__ == '__main__':
    main()
