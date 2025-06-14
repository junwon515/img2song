from transformers import LlavaForConditionalGeneration, LlavaProcessor
import torch

from core.utils import load_image_from_source, translate_to_english
from core.config import DEVICE, CAPTIONER_MODEL_NAME


class ImageCaptioner:
    def __init__(self):
        self.processor = LlavaProcessor.from_pretrained(CAPTIONER_MODEL_NAME)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            CAPTIONER_MODEL_NAME,
            device_map='auto' if DEVICE == 'cuda' else 'cpu',   
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
        )

    def generate(self,
                 image_source: str,
                 prompt: str = 'USER: What is the overall mood of this image, and what story does it tell? Please answer in English only! ASSISTANT:',
                 max_new_tokens: int = 100
                 ) -> str:
        try:
            image = load_image_from_source(image_source)
        except Exception as e:
            raise ValueError(e)

        formatted_prompt = f'{prompt}\n<image>'
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
            
        decoded_text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_response = decoded_text.split('ASSISTANT:')[-1].strip()

        if '번역결과' in generated_response:
            title, text = generated_response.split('번역결과')
            generated_response = title + translate_to_english(text.strip())

        return generated_response
