import torch

def get_device(vram_threshold_gb : int = 4) -> str:
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

if __name__ == '__main__':
    device = get_device()
    print(f'Using device: {device}')