from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


class EmotionExtractor:
    def __init__(self, model_name: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest', device: str = 'cpu'):
        """
        감정 추출기 초기화

        Args:
            model_name (str): 사용할 사전 훈련된 모델 이름
            device (str): 'cpu' 또는 'cuda' (GPU 사용 시)
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.labels = [v for k, v in sorted(self.model.config.id2label.items())]

    def extract_emotion(self, text: str) -> dict:
        """
        텍스트로부터 감정을 추출

        Args:
            text (str): 입력 텍스트

        Returns:
            dict: {'label': str, 'scores': dict}
                label: 가장 높은 확률의 감정 레이블
                scores: 각 감정 레이블에 대한 확률 점수
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        top_label = max(scores, key=scores.get)
        return {'label': top_label, 'scores': scores}

if __name__ == '__main__':
    try:
        from util import get_device
        from img2story import Img2Story
        device = get_device()
        img2story = Img2Story(device=device)
        emotion_extractor = EmotionExtractor(device=device)
        while True:
            user_input = input('Enter image path: ')
            if user_input.lower() == 'exit':
                break

            generated_text = img2story.generate(user_input)
            print(f'Generated text: {generated_text}')
            emotion_result = emotion_extractor.extract_emotion(generated_text)
            print(f'Extracted emotion: {emotion_result["label"]}')
            print(f'Scores: {emotion_result["scores"]}')
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    except Exception as e:
        print(f'Error: {e}')