import streamlit as st
from clip_song_matcher.recommender import MusicRecommender


st.set_page_config(page_title='CLIP Music Recommender', layout='centered')
st.title('🎵 CLIP 기반 음악 추천기')

recommender = MusicRecommender()

mode = st.radio('입력 방식 선택', ['이미지로 추천', '텍스트로 추천'])

top_k = st.slider('추천 개수 선택', min_value=1, max_value=10, value=5)

if mode == '이미지로 추천':
    image_input = st.file_uploader('이미지를 업로드하거나 URL을 입력하세요', type=['png', 'jpg', 'jpeg'])
    image_url = st.text_input('이미지 URL을 입력하세요 (선택 사항)', placeholder='https://example.com/image.jpg')

    if (image_input or image_url) and st.button('🎧 음악 추천 받기'):
        path = image_input if image_input else image_url
        try:
            results = recommender.recommend_image(path, top_k=top_k)

            st.subheader(f'Top {top_k} 추천 결과:')
            for idx, (id, url, sim) in enumerate(results, start=1):
                st.markdown(f'**{idx}.** `{id}` | 유사도: `{sim:.4f}`')
                st.markdown(f'[🔗 YouTube 링크]({url})')

        except Exception as e:
            st.error(f'추천 중 오류가 발생했습니다: {e}')
            st.warning('이미지 URL이 유효한지 확인하거나 이미지를 다시 업로드해 주세요.')

elif mode == '텍스트로 추천':
    query = st.text_area('감정, 분위기 또는 가사 조각을 입력하세요', height=100)

    if query and st.button('🎧 음악 추천 받기'):
        results = recommender.recommend_text(query, top_k=top_k)

        st.subheader(f'Top {top_k} 추천 결과:')
        for idx, (id, url, sim) in enumerate(results, start=1):
            st.markdown(f'**{idx}.** `{id}` | 유사도: `{sim:.4f}`')
            st.markdown(f'[🔗 YouTube 링크]({url})')
