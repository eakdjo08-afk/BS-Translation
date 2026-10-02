"""OpenAI를 이용한 다국어 번역 로직."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LANGUAGES = {"en": "English", "ja": "Japanese", "vi": "Vietnamese"}
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "low"
MAX_CHARS = 5000
# 5,000자 번역도 넉넉히 끝나도록 잡은 요청당 제한 시간(초)
REQUEST_TIMEOUT = 120

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


def _request_options() -> dict:
    """모델별로 지원 여부가 다른 선택 파라미터를 설정값에서 만든다."""
    options = {}
    # gpt-6-astra 등 일부 모델은 temperature 기본값(1)만 허용하므로,
    # OPENAI_TEMPERATURE가 설정된 경우에만 보낸다.
    temperature = _get_config("OPENAI_TEMPERATURE")
    if temperature not in (None, ""):
        try:
            options["temperature"] = float(temperature)
        except ValueError:
            raise TranslationError(
                "OPENAI_TEMPERATURE 값이 올바르지 않습니다. 숫자로 입력하거나 비워 두세요."
            )
    # 추론 강도를 낮추면 응답이 크게 빨라진다. 설정하지 않으면 "low"를 쓰고,
    # 빈 값으로 설정하면 보내지 않는다 (reasoning_effort를 지원하지 않는 모델용).
    effort = _get_config("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    if effort:
        options["reasoning_effort"] = effort
    return options


def translate(text: str, targets: list[str], model: str | None = None) -> dict[str, str]:
    """text를 targets 언어들로 번역한다. 언어별 요청을 동시에 보내 응답 시간을 줄인다.

    실패하면 사용자용 메시지를 담은 TranslationError를 발생시킨다.
    """
    api_key = _get_config("OPENAI_API_KEY")
    if not api_key:
        raise TranslationError(
            "OpenAI API 키가 설정되지 않았습니다. `.env` 또는 Secrets를 확인하세요."
        )

    # 설정값은 작업 스레드가 아닌 여기서 한 번만 읽는다.
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=1)
    model = model or get_model()
    options = _request_options()

    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = {
            code: executor.submit(_translate_one, client, model, options, text, code)
            for code in targets
        }
        # 하나라도 실패하면 그 오류를 그대로 사용자에게 보여준다.
        return {code: future.result() for code, future in futures.items()}


def _translate_one(client: OpenAI, model: str, options: dict, text: str, code: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            **options,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f'Target languages: "{code}" ({LANGUAGES[code]})\n\nText:\n{text}',
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
    except openai.APITimeoutError:
        raise TranslationError(
            "번역 시간이 너무 오래 걸립니다. 글을 나눠서 다시 시도해 주세요."
        )
    except openai.APIConnectionError:
        raise TranslationError(
            "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."
        )
    except openai.BadRequestError as e:
        raise TranslationError(
            f"요청 설정이 올바르지 않습니다. 모델 설정을 확인하세요: {_summarize(e)}"
        )
    except openai.OpenAIError as e:
        raise TranslationError(f"번역 중 오류가 발생했습니다: {_summarize(e)}")

    try:
        data = json.loads(response.choices[0].message.content or "")
        return str(data[code])
    except (json.JSONDecodeError, KeyError, TypeError):
        raise TranslationError(
            "번역 결과를 해석하지 못했습니다. 잠시 후 다시 시도해 주세요."
        )


def _summarize(e: Exception, limit: int = 150) -> str:
    """OpenAI 오류에서 응답 본문 대신 사람이 읽을 메시지만 꺼낸다."""
    body = getattr(e, "body", None)
    message = None
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = error.get("message")
    if not message:
        message = getattr(e, "message", None) or str(e)
    message = " ".join(str(message).split())
    return message if len(message) <= limit else message[: limit - 1].rstrip() + "…"
