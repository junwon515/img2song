from transformers import CLIPTokenizer

from core.config import CLIP_MODEL_NAME


class TokenChecker:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.max_length = 77

    def check_token_length(self, text: str) -> int:
        tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)
        return len(tokens['input_ids'])

    def check(self, text: str) -> bool:
        return self.check_token_length(text) <= self.max_length
