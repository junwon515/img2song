# 🎧 Img2Song

멀티모달(이미지/텍스트) 입력으로 음악을 추천해주는 CLIP 기반 추천 시스템입니다.  
이미지나 텍스트(감정, 분위기, 가사 등)를 입력하면, 유사한 음악을 추천해줍니다.

---

## 🧠 주요 기능

- 🎼 이미지 또는 텍스트로 음악 추천
- 🧠 CLIP 임베딩을 projection head로 fine-tuning
- 📺 YouTube에서 음악 데이터 자동 수집 및 전처리
- 🔁 전체 데이터 파이프라인 CLI 제공
- 🌐 Streamlit UI 또는 CLI 둘 다 지원

---

## 🔧 설치 방법

```bash
# 프로젝트 클론
git clone https://github.com/junwon515/img2song.git
cd img2song

# 가상환경 생성
python -m venv .venv
.venv\Scripts\activate

# 의존성 설치
pip install yt_dlp pillow librosa webvtt-py tqdm streamlit
pip install transformers accelerate bitsandbytes
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

---

## 📁 프로젝트 구조

```
clip_song_prepper/      # 데이터 수집 및 전처리 (YouTube 기반)
clip_song_matcher/      # 모델 학습 및 음악 추천 로직
core/                   # 공통 설정 및 유틸 함수
app.py                  # UI 실행용 streamlit 앱
```

---

## 📦 1. 데이터 수집 및 전처리

### 유튜브 링크 추가 및 관리
```bash
# YouTube 링크 추가
python -m clip_song_prepper.main --step add --url "<YouTube URL>" --title "Lofi Beats"

# 링크 리스트 보기
python -m clip_song_prepper.main --step list

# 링크 삭제
python -m clip_song_prepper.main --step remove --id "<YouTube ID>"
```

### 전체 파이프라인 실행
```bash
# 전체 실행
python -m clip_song_prepper.main --step all

# 개별 단계 실행도 가능
python -m clip_song_prepper.main --step fetch       # YouTube 메타데이터 수집
python -m clip_song_prepper.main --step fetch --url "<YouTube URL>" # 개별도 가능
python -m clip_song_prepper.main --step caption     # 이미지 캡셔닝
python -m clip_song_prepper.main --step preprocess  # 텍스트 전처리
python -m clip_song_prepper.main --step embed       # CLIP 임베딩 생성
```

---

## 🧪 2. 모델 학습 (Projection Head)

```bash
python -m clip_song_matcher.main train
```

옵션을 커스터마이즈 하고 싶다면:
```bash
python -m clip_song_matcher.main train `
  --npz_path ./data/embed.npz `
  --input_dim 512 `
  --proj_dim 128 `
  --lr 1e-4 `
  --batch_size 64 `
  --epochs 20 `
  --save_path ./checkpoints/proj_head.pt
```

---

## 🎧 3. 음악 추천 사용법

### CLI 기반 추천

```bash
# 이미지 기반 추천
python -m clip_song_matcher.main image ./examples/cover.jpg --top_k 5

# 텍스트 기반 추천
python -m clip_song_matcher.main text "잔잔하고 감성적인 피아노곡" --top_k 5
```

> 텍스트는 자동으로 영어로 번역되어 CLIP에 입력됩니다.

---

## 🌐 4. Streamlit UI 실행

```bash
streamlit run app.py
```

> 이미지 업로드 또는 텍스트 입력을 통해 간편하게 음악을 추천받을 수 있습니다.

---

## 📁 데이터 파일 설명

| 경로 | 설명 |
|------|------|
| `data/youtube_urls.json` | 수집 대상 YouTube 링크 목록 |
| `data/metadata.json` | 수집된 유튜브 메타데이터 (썸네일, 자막 등) |
| `data/embeddings.npz` | 전처리된 CLIP 임베딩 |
| `checkpoints/` | 학습된 projection head 보관 폴더 |

---

## 📚 기술 스택

- **Multimodal Vision-Language Model**
  - CLIP (OpenAI)
  - LLaVA (Meta)
- **PyTorch** / **HuggingFace Transformers**
- **YouTube Metadata** (`yt_dlp`)
- **Streamlit** (UI)
- **Google Translate API**
