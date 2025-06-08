from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import torch
import os


class Img2Story:
    def __init__(self, model_name: str = 'unsloth/llava-1.5-7b-hf-bnb-4bit' , device: str = 'cpu'):  
        """
        Img2Story 클래스 초기화.
        Llava 모델을 로드하고, 이미지와 텍스트를 처리하기 위한 프로세서를 설정합니다.
        
        Args:
            model_name (str): 사용할 Llava 모델의 Hugging Face ID.
                              기본값은 'unsloth/llava-1.5-7b-hf-bnb-4bit'입니다.
            device (str): 사용할 디바이스('cuda' 또는 'cpu'). 기본값은 'cpu'입니다.
        """
        self.model_name = model_name
        self.device = device

        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map='auto' if self.device == 'cuda' else 'cpu',   
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
        )

    def generate(self,
                 image_path: str,
                 prompt: str = 'USER: What is the overall mood of this image, and what story does it tell? ASSISTANT:',
                 max_new_tokens: int = 512
                 ) -> str:
        """
        주어진 이미지 경로와 프롬프트를 사용하여 텍스트를 생성합니다.
        
        Args:
            image_path (str): 이미지 파일의 경로.
            prompt (str): 모델에 전달할 프롬프트.
                          LLaVA의 대화형 프롬프트 형식에 맞춰 'USER: ... ASSISTANT:'를 사용하는 것이 좋습니다.
            max_new_tokens (int): 생성할 최대 새 토큰 수.

        Returns:
            str: 생성된 텍스트.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'Image file does not exist at path: {image_path}')

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f'Error opening image: {e}')

        formatted_prompt = f'{prompt}\n<image>'
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
            
        decoded_text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_end_index = decoded_text.find('ASSISTANT:')
        if prompt_end_index != -1:
            generated_response = decoded_text[prompt_end_index + len('ASSISTANT:'):].strip()
        else:
            generated_response = decoded_text

        return generated_response
    
if __name__ == '__main__':
    try:
        from util import get_device
        img2story = Img2Story(device=get_device())
        while True:
            user_input = input('Enter image path: ')
            if user_input.lower() == 'exit':
                break

            generated_text = img2story.generate(user_input)
            print(f'Generated text: {generated_text}')
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    except Exception as e:
        print(f'Error: {e}')