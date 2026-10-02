import streamlit as st

from translator import (
    LANGUAGES,
    MAX_CHARS,
    TranslationError,
    get_model,
    has_api_key,
    translate,
)

# 언어 코드 → (국기, 원어 이름, 한국어 이름)
LANGUAGE_DISPLAY = {
    "en": ("🇺🇸", "English", "영어"),
    "ja": ("🇯🇵", "日本語", "일본어"),
    "vi": ("🇻🇳", "Tiếng Việt", "베트남어"),
}

st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="wide")

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&family=Noto+Sans+KR:wght@400;600;700&family=Noto+Sans+JP:wght@400;600;700&family=Noto+Color+Emoji&display=swap");
    .block-container { max-width: 1200px; padding-top: 2.5rem; }
    .hero h1 { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.25rem; padding: 0; }
    .hero p { font-size: 1.05rem; opacity: 0.7; margin: 0 0 1.25rem; }
    .char-count { text-align: right; font-size: 0.85rem; opacity: 0.6; margin-top: -0.6rem; }
    .char-count.near-limit { color: #E5484D; opacity: 1; font-weight: 600; }
    .card-title { font-size: 1.15rem; font-weight: 700; margin: 0; }
    .card-sub { font-size: 0.85rem; opacity: 0.6; margin: 0 0 0.25rem; }
    .empty-state { text-align: center; padding: 2.5rem 1rem; opacity: 0.55; }
    .empty-state .icon { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
    /* 일본어 결과는 일본어 자형(한자)으로 표시한다. */
    .st-key-result-ja pre, .st-key-result-ja code {
        font-family: "Noto Sans JP", "Noto Sans", sans-serif !important;
    }
    .st-key-results pre { line-height: 1.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("input_text", "")
st.session_state.setdefault("targets", list(LANGUAGES))
st.session_state.setdefault("results", None)


@st.cache_data(show_spinner=False)
def cached_translate(text: str, targets: tuple[str, ...], model: str) -> dict[str, str]:
    # model을 캐시 키에 포함해, 모델을 바꾸면 새로 번역한다.
    return translate(text, list(targets), model)


def reset() -> None:
    st.session_state.input_text = ""
    st.session_state.results = None


def language_label(code: str) -> str:
    flag, native, korean = LANGUAGE_DISPLAY[code]
    return f"{flag} {native} · {korean}"


with st.sidebar:
    st.subheader("⚙️ 설정")
    st.markdown(f"**사용 모델**  \n`{get_model()}`")
    if not has_api_key():
        st.warning("OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Secrets를 확인하세요.")
    st.divider()
    st.subheader("📖 사용 방법")
    st.markdown(
        "1. 번역할 글을 입력하세요. 원문 언어는 자동으로 감지됩니다.\n"
        "2. 번역할 언어를 고르고 **번역하기**를 누르세요.\n"
        "3. 결과 오른쪽 위의 복사 버튼으로 바로 복사하세요."
    )

st.markdown(
    '<div class="hero"><h1>🌐 다국어 번역기</h1>'
    "<p>글을 입력하면 영어 · 일본어 · 베트남어로 번역합니다</p></div>",
    unsafe_allow_html=True,
)

st.text_area(
    "번역할 글",
    key="input_text",
    placeholder="번역할 글을 입력하세요...",
    height=220,
    max_chars=MAX_CHARS,
)
char_count = len(st.session_state.input_text)
near_limit = " near-limit" if char_count >= MAX_CHARS * 0.9 else ""
st.markdown(
    f'<div class="char-count{near_limit}">{char_count:,} / {MAX_CHARS:,}자</div>',
    unsafe_allow_html=True,
)

st.pills(
    "번역 언어",
    options=list(LANGUAGES),
    selection_mode="multi",
    format_func=language_label,
    key="targets",
)

with st.container(horizontal=True):
    translate_clicked = st.button("번역하기", type="primary", icon=":material/translate:")
    st.button("초기화", on_click=reset, icon=":material/restart_alt:")

if translate_clicked:
    text = st.session_state.input_text.strip()
    targets = [code for code in LANGUAGES if code in st.session_state.targets]

    if not text:
        st.warning("번역할 글을 입력해 주세요.")
    elif not targets:
        st.warning("번역할 언어를 하나 이상 선택해 주세요.")
    elif len(text) > MAX_CHARS:
        st.warning(f"최대 {MAX_CHARS:,}자까지 입력할 수 있습니다.")
    else:
        try:
            with st.spinner("번역 중..."):
                st.session_state.results = cached_translate(text, tuple(targets), get_model())
        except TranslationError as e:
            # 이전 결과가 새 입력의 번역으로 오해되지 않도록 지운다.
            st.session_state.results = None
            st.error(str(e))

st.divider()
st.subheader("번역 결과")

results = st.session_state.results
with st.container(key="results"):
    if results:
        # 좁은 화면에서는 Streamlit이 컬럼을 자동으로 세로로 쌓는다.
        for code, column in zip(results, st.columns(len(results))):
            flag, native, korean = LANGUAGE_DISPLAY[code]
            with column, st.container(border=True, key=f"result-{code}"):
                st.markdown(
                    f'<p class="card-title">{flag} {native}</p><p class="card-sub">{korean}</p>',
                    unsafe_allow_html=True,
                )
                st.code(results[code], language=None, wrap_lines=True)
    else:
        with st.container(border=True):
            st.markdown(
                '<div class="empty-state"><span class="icon">💬</span>'
                "번역 결과가 여기에 표시됩니다</div>",
                unsafe_allow_html=True,
            )
