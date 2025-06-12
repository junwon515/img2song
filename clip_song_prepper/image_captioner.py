from transformers import LlavaForConditionalGeneration, LlavaProcessor
import torch

from core.utils import load_image_from_source
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
                 prompt: str = 'USER: What is the overall mood of this image, and what story does it tell? ASSISTANT:',
                 max_new_tokens: int = 100
                 ) -> str:
        """
        이미지 경로와 프롬프트를 사용하여 스토리를 생성

        Args:
            image_source (str): 이미지 파일 경로 또는 URL
            prompt (str): 생성할 스토리에 대한 프롬프트 텍스트
                          기본값은 'USER: What is the overall mood of this image, and what story does it tell? ASSISTANT:'
            max_new_tokens (int): 생성할 최대 토큰 수, 기본값은 100
            
        Returns:
            str: 생성된 스토리 텍스트
        """
        try:
            image = load_image_from_source(image_source)
        except Exception as e:
            raise ValueError(e)

        formatted_prompt = f'{prompt}\n<image>'
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
            
        decoded_text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_response = decoded_text.split('ASSISTANT:')[-1].strip()

        return generated_response
