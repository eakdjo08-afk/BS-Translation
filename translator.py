"""OpenAI를 이용한 다국어 번역 로직."""

import json
import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LANGUAGES = {"en": "English", "ja": "Japanese", "vi": "Vietnamese"}
DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000

SYSTEM_PROMPT = (
    "You are a professional translator. Detect the source language automatically "
    "and translate the user's text into the requested languages. Preserve meaning, "
    "tone, line breaks, and formatting. If the text is already in a target language, "
    "return it lightly polished in that language. Do not add explanations. Respond "
    "only in JSON with keys for each requested language code."
)


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 담은 번역 오류."""


def _get_config(key: str, default: str | None = None) -> str | None:
    """st.secrets를 먼저 확인하고, 없으면 환경 변수(.env)에서 읽는다."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # secrets.toml이 없거나 Streamlit 밖에서 실행된 경우
        pass
    return os.getenv(key, default)


def get_model() -> str:
    return _get_config("OPENAI_MODEL") or DEFAULT_MODEL


def has_api_key() -> bool:
    return bool(_get_config("OPENAI_API_KEY"))


def translate(text: str, targets: list[str], model: str | None = None) -> dict[str, str]:
    """text를 targets 언어 코드들로 한 번의 API 호출로 번역한다.

    실패하면 사용자용 메시지를 담은 TranslationError를 발생시킨다.
    """
    api_key = _get_config("OPENAI_API_KEY")
    if not api_key:
        raise TranslationError(
            "OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Secrets를 확인하세요."
        )

    requested = ", ".join(f'"{code}" ({LANGUAGES[code]})' for code in targets)
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model or get_model(),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Target languages: {requested}\n\nText:\n{text}",
                },
            ],
        )
    except openai.AuthenticationError:
        raise TranslationError(
            "OpenAI API 키가 올바르지 않습니다. `.env` 또는 Secrets를 확인하세요."
        )
    except openai.RateLimitError:
        raise TranslationError("요청이 많습니다. 잠시 후 다시 시도해 주세요.")
    except openai.NotFoundError:
        raise TranslationError(
            "모델을 찾을 수 없습니다. `OPENAI_MODEL` 값을 확인하세요."
        )
    except openai.APIConnectionError:
        # APITimeoutError도 여기에 포함된다.
        raise TranslationError(
            "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."
        )
    except openai.OpenAIError as e:
        raise TranslationError(f"번역 중 오류가 발생했습니다: {_summarize(e)}")

    try:
        data = json.loads(response.choices[0].message.content or "")
        results = {code: str(data[code]) for code in targets}
    except (json.JSONDecodeError, KeyError, TypeError):
        raise TranslationError(
            "번역 결과를 해석하지 못했습니다. 잠시 후 다시 시도해 주세요."
        )
    return results


def _summarize(e: Exception) -> str:
    message = getattr(e, "message", None) or str(e)
    return message.splitlines()[0][:200]
