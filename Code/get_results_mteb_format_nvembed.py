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
# from mteb.tasks.Retrieval.multilingual.WikipediaRetrievalMultilingual import WikipediaRetrievalMultilingual
# from mteb.tasks.Retrieval.multilingual.XPQARetrieval import XPQARetrieval
# from mteb.tasks.Retrieval.multilingual.XQuADRetrieval import XQuADRetrieval
# from mteb.tasks import FiQA2018 , ArguAna, SciFact, SCIDOCS, TRECCOVID, Touche2020
#CHANGED THE MULTILINGUAL DOC FILE TO RUN ONLY HINDI..REMEMBER TO REVERT BACK
# from mteb.tasks import MultiLongDocRetrieval
# from mteb.tasks import FiQA2018
# from mteb.tasks import MIRACLRetrieval
#CHNAGES THE FILE TO RUN FOR ONLY HINDI AND CHANGED THE NUM_SAMPLES FROM 18585 TO 1547 FOR INDICQARETRIEVAL...REMEMEMVER TO REVERT BACK
# from mteb.tasks import IndicQARetrieval
parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='nvidia/NV-Embed-v2',
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
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True, cache_dir='../cache_dir')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True,  cache_dir='../cache_dir')
        self.prompt = "Retrieve the relevant paragraph that provides an answer to the given medical query."
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        # print(queries)
        # if args.prefix_type == 'query_or_passage':
        input_texts = [str(q['txt']) for q in queries]
        # else:
            # input_texts = ['self.prompt{}'.format(t) for t in queries]

        return self._do_encode(input_texts,max_length=512,is_query=True)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        # if args.doc_as_query:
        #     return self.encode_queries([d['text'] for d in corpus], **kwargs)

        # input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = [str(doc['txt']).strip() for doc in corpus]
        # # no need to add prefix for instruct models
        # if args.prefix_type == 'query_or_passage':
        #     input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts,max_length=512,is_query=False)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[dict],max_length:int, is_query:bool) -> np.ndarray:
        encoded_embeds = []
        batch_size = 100* self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[dict] = input_texts[start_idx: start_idx + batch_size]
            # print(batch_input_texts)

            # batch_dict = create_batch_dict(self.tokenizer, batch_input_texts,max_length=max_length)
            # batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                # outputs: BaseModelOutput = self.encoder(**batch_dict)
                # embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                # print(batch_dict.keys())
                
                if is_query==True:
                     embeds=self.encoder.encode(batch_input_texts, instruction=self.prompt, max_length=max_length)
                else:
                     embeds=self.encoder.encode(batch_input_texts, instruction='', max_length=max_length)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    # assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)
    # print
    model = RetrievalModel()
#     PUBMED_ROOT = '/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/PubMedVision/PubMedVQA'
# #     folders = [#'Pelvic_cavity',
# #               # 'Endoscopy',
# #         'Digital_health', 'Oral_cavity', 'Digital_Amplified', 'Cancer_therapy_of_older_adults', 'Brain', 'Cancer_Treatment', 'C-Spine-Routine', 'Digital_Healthcare', 'Abdomen', 'Eye', 'Computed_Tomography', 'Computed_tomography', 'Magnetic_Resonance_Imaging', 'Orbita', 'MicroNano_motor', 'Others', 'Cell', 'Microfluidic_Chip', 'Digital_Transformation', 'Gastrointestinal_tract', 'Digital_Content', 'Chest_Breast', 'Lower_Upper_Limb_Foot', 'Ultrasound', 'C-Spine', 'Digital_Dermatitis_Lesion_Score'
    
# # ]
#     folders = [
#         # 'Abdomen',
#         # 'Brain',
#         # 'Chest',
#         'Cell',
#         'Others',
#         'Lower_Upper_Limb_Foot',
#         'Oral_cavity',
#         'Eye',
#         'Chest_Breast',
#         'Gastrointestinal_tract',
#         'Pelvic_cavity'
    
# ]
    ROOT = '/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/Relevant_Articles_Ret_MultiCare'
    folders=[
        # "Cardiology",
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
        "Psychiatry and Behavioral Health",
        "Pulmonology",
        "Rheumatology and Immunology"
    ]
 #   folders=[
# 'Pharmacology',
#  'Surgical_Specialties',
#  'Neurology_and_Neuroscience',
 # 'Miscellaneous',
 # 'Hematology',
 # 'Ophthalmology_and_Sensory_Systems',
 # 'Microbiology_and_Cell_Biology',
 # 'Orthopedics_and_Musculoskeletal',
 # 'Cardiology',
 # 'Endocrinology_and_Diabetes',
 # 'Radiology_and_Imaging',
 # 'Skeletal_System',
 # 'Psychiatry_and_Mental Health',
 # 'Respiratory_and_Pulmonology',
 # 'Anatomy and Physiology',
 # 'Gastroenterology',
 # 'Reproductive_System',
 # 'Dermatology'
 #   'MMQSD_Ret'
#]
    for name in folders:
        class Custom_Class(AbsTaskRetrieval):
            metadata = TaskMetadata(
                name=f"{name}",
                model_type="nv_embed",
                dataset={
                    "path": os.path.join(ROOT, name),
                    "revision": "None",
                    "model_type":"nv_embed",
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
        results_dict = evaluation.run(model=model,output_folder=f'./results_new/MultiCare/nv_embed/{name}',eval_splits=["test"],overwrite_results=True,corpus_chunk_size=400000)
        # , eval_splits=["test"], output_folder='./hindi_results/bge-m3', overwrite_results=eval_args.overwrite, corpus_chunk_size=200000
        
    
        print(results_dict)
        
        print("==================================================")
        print("Finish evaluation for model:")



if __name__ == '__main__':
    main()