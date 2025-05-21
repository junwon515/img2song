# 🎵 Img2Song: 이미지 기반 음악 생성 모델

**Img2Song**은 이미지(예: 유튜브 썸네일)를 입력으로 받아 음악을 생성하는 인공지능 모델입니다.  
YouTube에서 썸네일과 오디오 데이터를 수집하고, CNN 기반 인코더-디코더 모델을 학습하여 이미지-음악 쌍을 생성합니다.

---

## ✨ 주요 기능

- YouTube 플레이리스트에서 이미지 & 오디오(10초) 자동 수집
- 썸네일과 오디오의 자동 전처리
- CNN 기반 인코더-디코더 모델 학습
- 향후 이미지 입력 → 음악 생성 기능 구현 가능

---

## 🛠️ 설치 방법

### 1. 파이썬 환경 준비

Python 3.9 이상이 설치되어 있어야 합니다.

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows 기준
```

### 2. 필수 라이브러리 설치

아래 명령어를 순서대로 실행하세요:

```bash
pip install torch torchaudio librosa matplotlib opencv-python tqdm yt-dlp
```

### 3. FFmpeg 설치 (Windows 기준)

yt-dlp와 librosa는 FFmpeg가 필요합니다.

- FFmpeg 공식 사이트 접속: https://ffmpeg.org/download.html
- Windows용 zip 다운로드 (예: ffmpeg-release-essentials.zip)
- 압축 해제 후 bin 폴더 경로 복사 (예: C:\ffmpeg\bin)
- Windows 시작 메뉴 > "환경 변수 편집" > 시스템 변수 Path에 해당 경로 추가
- 설치 확인:

```bash
ffmpeg -version
```

---

## 📥 YouTube에서 썸네일 + 오디오 수집

작업 폴더에서 data 폴더를 생성합니다.

```bash
mkdir data
```

YouTube 재생목록에서 이미지와 오디오(10초)를 수집합니다.  
아래 명령어를 복사해서 붙여 넣고, 마지막 줄의 플레이리스트 URL을 원하는 것으로 바꾸세요.

```bash
yt-dlp --yes-playlist `
  --write-thumbnail `
  --extract-audio --audio-format wav `
  --postprocessor-args "-ss 0 -t 10" `
  --output "data/%(id)s.%(ext)s" `
  https://www.youtube.com/playlist?list=<여기에_플레이리스트_ID_입력>
```

여러 개의 플레이리스트를 사용하려면 위 명령어를 플레이리스트 URL만 바꿔 여러 번 실행하세요.

---

## 🧹 데이터 전처리

학습에 사용할 이미지와 오디오 데이터를 모델에 맞게 변환하고 정리하는 과정입니다.  
`preprocess.py` 스크립트를 실행하여 수집한 원본 데이터를 전처리하세요.

```bash
python preprocess.py
```

---

## 🧠 모델 학습

모델 학습, 전처리, 모델 정의는 모두 img2mel_net.py에 통합되어 있습니다.
data/ 폴더 내 이미지-오디오 쌍을 기반으로 학습이 진행됩니다.

```bash
$ python img2mel_net.py
모드 선택 (train/generate): train
```

모델은 CNN 기반 인코더-디코더 구조로 구성되어 있으며, 학습이 완료되면 img2mel.pth로 저장됩니다.

---

## 📁 디렉터리 구조

```bash
img2song-model/
│
├── data/             # 썸네일 이미지 + 오디오 파일 + 전처리된 파일 (.jpg/.webp/.wav/.pt)
├── img2mel_net.py    # 모델 정의 + 전처리 + 학습 + 추론 통합 스크립트
├── preprocess.py     # 전처리 함수 모듈
└── README.md         # 설명서
```

---

## 📌 참고 사항

- 썸네일 이미지와 오디오 파일 이름은 YouTube 영상 ID 기준으로 자동 매칭됩니다.
- 오디오 길이는 10초로 고정되어 학습 안정성을 높입니다.
- GPU가 있다면 PyTorch에서 자동 인식하여 사용됩니다.
