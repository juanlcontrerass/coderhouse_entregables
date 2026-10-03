# Cliente de LLM robusto y asíncrono

Pre-entrega 1 (Coderhouse – AI Engineering): cliente de LLM asíncrono con una interfaz común para
**Gemini** (`google-genai`) y **Z.ai GLM** vía el router de HuggingFace (SDK `openai`).

## Estructura

| Archivo | Contenido |
|---|---|
| `llm_client/schemas.py` | `Provider`, `ChatMessage`, `LLMConfig`, `ModelResponse` (Pydantic) |
| `llm_client/base.py` | `BaseLLMClient`: contrato abstracto (`generate`, `generate_stream`) |
| `llm_client/gemini_client.py` | `GeminiClient` |
| `llm_client/zai_client.py` | `ZaiClient` |
| `llm_client/retry.py` | Reintentos con backoff exponencial |
| `llm_client/manager.py` | `AsyncLLMManager` (factory) |
| `main.py` | Script de validación |

## Uso (en un ambiente venv con Python 3.12)

```bash
pip install -r requirements.txt
copy .env.example .env   # y completar GOOGLE_API_KEY y HF_TOKEN
python main.py
```
