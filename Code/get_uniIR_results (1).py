import os
import argparse
from omegaconf import OmegaConf
import tqdm
import gc
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.cuda.amp import autocast
import transformers

from uniir_utils import build_model_from_config, set_seed

from PIL import Image
import torch
from omegaconf import OmegaConf
import argparse
from clip_sf import CLIPScoreFusion
from blip_ff import BLIPFeatureFusion
from PIL import Image

# model=CLIPScoreFusion()
# model.load_state_dict(torch.load("./UniIR_Checkpoints/clip_sf_large.pth")["model"])
# tokenizer=
# text="Hello How are You?"
# text_tensor=clip.toke

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

print("Import Done")
parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='./UniIR_Checkpoints/clip_sf_large.pth',
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


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder =CLIPScoreFusion()
        self.encoder.load_state_dict(torch.load(args.model_name_or_path,weights_only=False)["model"])
        self.tokenizer=self.encoder.get_tokenizer()
        self.img_processor=self.encoder.get_img_preprocess_fn()
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        # print(self.encoder.text_encoder.config)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [str(q['txt']) for q in queries]
        input_images=[q['img'] for q in queries]

        return self._do_encode(input_texts,input_images,max_length=512,is_query=True)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = [str(doc['txt']).strip() for doc in corpus]
        input_images=[doc['img'] for doc in corpus] #Only of IT2I

        return self._do_encode(input_texts, input_images,max_length=1,is_query=True)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[dict],input_images,max_length:int, is_query:bool) -> np.ndarray:
        encoded_embeds = []
        batch_size = 1* self.gpu_count
        if is_query:
            batch_size=1
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[dict] = input_texts[start_idx: start_idx + batch_size]
            if input_images is not None:
                batch_input_images= input_images[start_idx: start_idx + batch_size]

            with torch.cuda.amp.autocast():
                text=batch_input_texts[0]
                if is_query==True:
                    images=batch_input_images[0]
                    image_tensor=self.img_processor(images).unsqueeze(0).to('cuda')
                    # print(image_tensor.shape)
                    image_embeddings=self.encoder.encode_image(image_tensor)
                    text_tensors=self.tokenizer(batch_input_texts).to('cuda')
                    # print(text_tensors.keys())
                    text_embeddings=self.encoder.encode_text(text_tensors)
                    # embeds=self.encoder.encode_multimodal_input(text_tensors, image_tensor)
                    embeds=image_embeddings+text_embeddings
                else:
                    encoder_hidden_states = torch.zeros(batch_size, 197, 1024).to('cuda')
                    encoder_attention_mask = torch.zeros(batch_size, 197).to('cuda')  # Shape: (batch_size, seq_len)
                    text_tensors=self.tokenizer(batch_input_texts).to('cuda')
                    # embeds=self.encoder.text_encoder(input_ids=text_tensors['input_ids'],
                    #                                  attention_mask=text_tensors['attention_mask'],
                    #                                  encoder_hidden_states=encoder_hidden_states,
                    #                                  encoder_attention_mask=encoder_attention_mask).pooler_output
                    embeds=self.encoder.encode_text(text_tensors)
                    
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    # assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)
    # print
    model = RetrievalModel()
    # ROOT = '/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/Relevant_Articles_Ret_MultiCare'
    # folders=[
    #     "Cardiology",
    #     "Dermatology",
    #     "Endocrinology",
    #     "Gastroenterology",
    #     "Genetics and Genomics",
    #     "Hematology",
    #     "Infectious Diseases",
    #     "Neurology",
    #     "Obstetrics and Gynecology",
    #     "Oncology",
    #     "Orthopedics",
    #     "Pathology",
    #     "Psychiatry and Behavioral Health",
    #     "Pulmonology",
    #     "Rheumatology and Immunology"
    # ]
    ROOT='clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I'
    folders= [
        "Abdomen",
              # "Head", "Reproductive and Urinary System", "Spine and Muscles", "Thorax"
    ]
    for name in folders:
        class Custom_Class(AbsTaskRetrieval):
            metadata = TaskMetadata(
                name=f"{name}",
                dataset={
                    "path": f"{ROOT}/{name}",
                    "revision": "None",
                    "model_type":"CLIP_SF",
                },
                description=(
                    "arguana, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
                    " prediction, to document classification and recommendation."
                ),
                reference="https://allenai.org/data/scidocs",
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
        results_dict = evaluation.run(model=model,output_folder=f'./results_new/IT2I/CLIP_SF/{name}',eval_splits=["test"],overwrite_results=True,corpus_chunk_size=400000)
        # , eval_splits=["test"], output_folder='./hindi_results/bge-m3', overwrite_results=eval_args.overwrite, corpus_chunk_size=200000
        
    
        print(results_dict)
        
        print("==================================================")
        print("Finish evaluation for model:")



if __name__ == '__main__':
    main()
