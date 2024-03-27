import bentoml

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL.Image import Image
from pathlib import Path

import typing as t

PROMPT_TEMPLATE = """<image>\nUSER: What's the content of the image?\nASSISTANT:"""


@bentoml.service()
class BentoLlava:

    def __init__(self):
        self.model =LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    @bentoml.api()
    def generate(self, image: Image, prompt: str = PROMPT_TEMPLATE) -> str:
        # image = Image.open(image)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generate_ids = self.model.generate(**inputs, max_length=30)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output

# from PIL import Image
# import requests
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=prompt, images=image, return_tensors="pt")

# # Generate
# generate_ids = model.generate(**inputs, max_length=30)
# processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#t.Annotated[Path, bentoml.validators.ContentType('image/jpeg')],
