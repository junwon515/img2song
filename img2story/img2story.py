from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import torch
import os

class Img2Story:
    def __init__(self, model_name: str = 'unsloth/llava-1.5-7b-hf-bnb-4bit'):
        """
        Img2Story 클래스 초기화.
        Llava 모델을 로드하고, GPU 사용 가능 여부 및 VRAM을 확인하여 최적의 설정을 적용합니다.
        
        Args:
            model_name (str): 사용할 Llava 모델의 Hugging Face ID.
                              기본값은 'unsloth/llava-1.5-7b-hf-bnb-4bit'입니다.
        """
        self.model_name = model_name
        self.device = self._get_device()
        
        print(f'Loading Llava model: {self.model_name} on device: {self.device}')

        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map='auto' if self.device == 'cuda' else 'cpu',   
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
        )

    def _get_device(self, vram_threshold_gb : int = 4) -> str:
        """
        사용 가능한 디바이스(cuda 또는 cpu)를 확인하고 반환합니다.
        VRAM 임계값을 기준으로 GPU 사용 여부를 결정합니다.

        Args:
            vram_threshold_gb (int): VRAM 임계값 (GB). 기본값은 4GB입니다.

        Returns:
            str: 'cuda' 또는 'cpu'. 사용 가능한 디바이스를 반환합니다.
        """
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram >= vram_threshold_gb:
                return 'cuda'
            else:
                print(f'Insufficient VRAM ({vram:.2f} GB). Using CPU instead.')
        return 'cpu'

    def generate(self,
                 image_path: str,
                 prompt: str = 'USER: What is the overall mood of this image, and what story does it tell? ASSISTANT:',
                 max_new_tokens: int = 50
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

def main():
    """
    Img2Story 클래스의 인스턴스를 생성하고, 사용자로부터 이미지 경로를 입력받아
    해당 이미지에 대한 스토리와 무드를 생성하는 메인 함수입니다.
    """
    try:
        img2story = Img2Story()
    except Exception as e:
        print(f'Failed to initialize Img2Story: {e}')
        return

    while True:
        user_input = input('Enter image path: ')
        if user_input.lower() == 'exit':
            break

        try:
            generated_text = img2story.generate(user_input)
            print(f'\nGenerated text: {generated_text}\n')
        except FileNotFoundError as e:
            print(f'Error: {e}')
        except ValueError as e:
            print(f'Error processing image: {e}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')

if __name__ == '__main__':
    main()