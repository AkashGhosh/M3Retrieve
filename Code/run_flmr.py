import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from flmr import index_custom_collection
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval,FLMRTextConfig
from PIL import PngImagePlugin,ImageFile
import os
from PIL import Image, UnidentifiedImageError

from flmr import search_custom_collection, create_searcher
PngImagePlugin.MAX_TEXT_CHUNK = 1000000000 
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import tempfile
# tempfile.tempdir = "./temp_file_sys/pymp-1mcqzyam"
# folder_names=[
#     # "Cardiology",
#     # "Dermatology",
#     # "Endocrinology",
#     # "Gastroenterology",
#     # "Genetics and Genomics",
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

ROOT='/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I'
folder_names= [
    "Abdomen",
          "Head", "Reproductive and Urinary System", "Spine and Muscles", "Thorax"
] 

# def main():

#     # for name in folder_names:\
#     # path_to_folder=f'/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/Relevant_Articles_Ret_MultiCare/Cardiology'
#     passage_contents=pd.read_json('/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I/Abdomen/corpus.jsonl',lines=True)['_id'].astype(str).tolist()

#     image_paths=pd.read_json('/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I/Abdomen/corpus.jsonl',lines=True)['image_path'].astype(str).tolist()

#     base_image_path='/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I/'
#     image_paths=[base_image_path+item for  item in image_paths]
    
#     # custom_collection=passage_contents
#     custom_collection = [
#     (passage_content, None, image_path)
#     for passage_content, image_path in zip(passage_contents, image_paths)
# ]
    
#     index_custom_collection(
#         custom_collection=custom_collection,
#         model='./temp',
#         index_root_path=".",
#         index_experiment_name=f"IT2I",
#         # index_name=f"{path_to_folder.split('/')[-1]}",
#         index_name='IT2I',
#         nbits=8, # number of bits in compression
#         doc_maxlen=512, # maximum allowed document length
#         overwrite=True, # whether to overwrite existing indices
#         use_gpu=True, # whether to enable GPU indexing
#         indexing_batch_size=64,
#         model_temp_folder="tmp",
#         nranks=1, # number of GPUs used in indexing
#     )


# if __name__ == '__main__':
#     main()


checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-G"
image_processor_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

configuration = FLMRTextConfig()
query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
    checkpoint_path, subfolder="query_tokenizer", text_config=configuration)
context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
    checkpoint_path, subfolder="context_tokenizer", text_config=configuration)

# Load model and move it to GPU
model = FLMRModelForRetrieval.from_pretrained(
    checkpoint_path,
    query_tokenizer=query_tokenizer,
    context_tokenizer=context_tokenizer,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Set up the image processor
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

# ---------------------
# 2. Load and process queries and images
for folder in tqdm(folder_names):
    print(f'Processing {folder}...')
    path_to_folder = f'{ROOT}/{folder}'
    queries_df = pd.read_json(f'{path_to_folder}/queries.jsonl', lines=True)
    passage_contents=pd.read_json(f'{path_to_folder}/corpus.jsonl',lines=True)['text'].tolist()
    ids_list = queries_df['_id'].tolist()
    query_texts = [f"Retrieve the most relevant text to the given image and image caption: {caption}"
                   for caption in queries_df['caption'].tolist()]
    queries_image_paths = queries_df['image_path'].tolist()
    
    # Define a torchvision transform (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to tensor [C, H, W]
    ])
    
    # Open images, apply transforms, and stack them
    # image_tensors = [
    #     transform(Image.open(os.path.join('/home/kitsuchart/aakash/MedRAG/clustered_dataset_extended/PubMedVision', img_path[0])).convert("RGB"))
    #     for img_path in queries_image_paths
    # ]

    base_path = '/home/kitsuchart/aakash/MedRAG/mteb/clustered_dataset_extended/MedPix-2.0-main/MedPix-2-0/M3Retrieve_IT2I/'
    default_image_path = '/home/kitsuchart/aakash/MedRAG/proxy.jpg'

    
    def get_valid_image_path(image_path, default_image_path):
        """
        Returns a valid image path:
        - If image_path exists and is not corrupted, return it.
        - Otherwise, return the default_image_path.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            
            with Image.open(image_path) as img:
                img.verify()  # Ensure image is not corrupted
    
            return image_path  # Valid path and image
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"Warning: '{image_path}' is invalid or corrupted ({e}). Using default image.")
            return default_image_path

    
    # image_tensors = [
    #     transform(
    #         Image.open(
    #             os.path.join(base_path, img_path)
    #             if os.path.exists(os.path.join(base_path, img_path))
    #             else default_image_path
    #         ).convert("RGB")
    #     )
    #     for img_path in queries_image_paths
    # ]
    image_tensors = [
    transform(
        Image.open(
            get_valid_image_path(os.path.join(base_path, img_path), default_image_path)
        ).convert("RGB")
    )
    for img_path in queries_image_paths
]

    query_images = torch.stack(image_tensors)  # Shape: (num_queries, 3, 224, 224)
    
    # Process images with the image processor to obtain pixel values (batched)
    query_pixel_values = image_processor(query_images, return_tensors="pt")['pixel_values']
    
    # Tokenize the query texts (this returns batched tensors)
    query_encoding = query_tokenizer(query_texts)
    
    # Determine number of queries and desired batch size
    num_queries = len(query_texts)
    batch_size = 32  # or choose another appropriate batch size
    
    # ---------------------
    # 3. Batch-wise query embedding generation on GPU
    all_embeddings = []  # List to collect embeddings
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Loop over batches
        for i in tqdm(range(0, num_queries, batch_size)):
            batch_end = min(i + batch_size, num_queries)
            # Slice the tokenized inputs and image pixel values for the current batch
            batch_input_ids = query_encoding['input_ids'][i:batch_end].to(device)
            batch_attention_mask = query_encoding['attention_mask'][i:batch_end].to(device)
            batch_pixel_values = query_pixel_values[i:batch_end].to(device)
            
            inputs = dict(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
            )
            
            # Run the model's query encoder for this batch
            res = model.query(**inputs)
            # Assume that res.late_interaction_output contains the embeddings for the batch.
            # Move them to CPU for further processing (or keep on GPU if desired)
            all_embeddings.append(res.late_interaction_output.cpu())
    
    # Concatenate embeddings from all batches; final shape should be (num_queries, embedding_dim)
    query_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Create a dictionary mapping query IDs to query texts (optional)
    queries = {ids_list[i]: query_texts[i] for i in range(num_queries)}
    
    # ---------------------
    # 4. Perform retrieval using the generated query embeddings
    
    # Initiate a searcher (make sure the index has been built as per your earlier code)
    searcher = create_searcher(
        index_root_path=".",
        index_experiment_name=f"IT2I",
        index_name=f"IT2I",
        nbits=8,         # Number of bits in compression
        use_gpu=True,    # Use GPU for search if available
    )
    
    # Run the search on the custom collection
    ranking = search_custom_collection(
        searcher=searcher,
        queries=queries,
        query_embeddings=query_embeddings,
        num_document_to_retrieve=10,  # How many documents to retrieve per query
    )
    
    # ---------------------
    # 5. Analyze the retrieval results
    ranking_dict = ranking.todict()
    # print(ranking_dict)
    # Assume passage_contents is a list of document texts (loaded from your corpus)
    # For example:
    # passage_contents = pd.read_json(os.path.join(path_to_folder, 'corpus.jsonl'), lines=True)['text'].tolist()
    # (Uncomment and modify the following line as needed.)
    # passage_contents = ...
    
    all_results = []  # List to collect results for all queries
    
    for i in ids_list:
        retrieved_docs = ranking_dict[i]
        for doc in retrieved_docs:
            doc_idx = doc[0]         # Index of the retrieved document
            score = doc[2]           # Confidence score
            doc_text = passage_contents[doc_idx]  # Get text from corpus
            all_results.append({
                "query-id": i,
                "colbert_score": score,
                "corpus": doc_text,
            })
    
    # Create a DataFrame from the collected results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"retrieved_docs_IT2I_{folder}.csv", index=False)
    
    print(df_all)
