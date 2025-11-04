from transformers import AutoConfig, AutoModel, AutoImageProcessor, AutoTokenizer
import torch
from PIL import Image

checkpoint_path = "LinWeizheDragon/PreFLMR_ViT-L"
image_processor_name = "openai/clip-vit-large-patch14"
query_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer", trust_remote_code=True)
context_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, subfolder="context_tokenizer", trust_remote_code=True)

model = AutoModel.from_pretrained(checkpoint_path,
                                query_tokenizer=query_tokenizer,
                                context_tokenizer=context_tokenizer,
                                trust_remote_code=True,
                                )
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
text=['Hello How are you']
query_encoding = query_tokenizer(query_texts)
images=[Image.open('../proxy.jpg')]
query_pixel_values = image_processor(query_images, return_tensors="pt")['pixel_values']

inputs = dict(
    input_ids=query_encoding['input_ids'],
    attention_mask=query_encoding['attention_mask'],
    pixel_values=query_pixel_values,
)

# Run model query encoding
res = model.query(**inputs)

print(res.late_interaction_output)
