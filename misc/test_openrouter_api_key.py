"""
Quick diagnostic for OpenRouter API key issues.

OpenRouter is OpenAI-compatible: the OpenAI SDK works against it once you
point base_url at OpenRouter and read OPENROUTER_API_KEY instead of
OPENAI_API_KEY.

Usage:
    python tests/test_openrouter_api_key.py [MODEL]

Default MODEL is `openai/gpt-4o-mini` (cheap and ubiquitous on OpenRouter).
Pass any OpenRouter model id to test access to that specific one, e.g.:
    python tests/test_openrouter_api_key.py google/gemini-3-flash-preview
    python tests/test_openrouter_api_key.py deepseek/deepseek-v3.2
    python tests/test_openrouter_api_key.py openai/gpt-5.1

Note: OpenRouter native model ids do NOT include the `openrouter/` prefix
that LiteLLM uses.  LiteLLM's `openrouter/openai/gpt-5.1` is the same model
as OpenRouter's native `openai/gpt-5.1` — strip the leading `openrouter/`
when calling OpenRouter directly.

Checks, in order:
    1. OPENROUTER_API_KEY is set in the environment
    2. The key is accepted by the API (not invalid/revoked)
    3. The account has sufficient credits
    4. The requested model is accessible
"""
import os
import sys


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"

    # 1. Check env var
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("FAIL: OPENROUTER_API_KEY is not set.")
        print("  Fix: export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"OK: OPENROUTER_API_KEY is set ({masked})")

    # 2-4. Try a minimal API call against OpenRouter via the OpenAI SDK.
    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: 'openai' package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_completion_tokens=50,
        )
        # `content` can be None for reasoning models when the token budget
        # was consumed by reasoning, and on some providers when the model
        # returned no visible answer (tool calls, refusals normalized to
        # None, etc.).  Treat any successful response as a passing key
        # check — the auth + credits + model-id all worked if we got here.
        text = ""
        if response.choices:
            content = response.choices[0].message.content
            if content is not None:
                text = content.strip()
        if text:
            print(f"OK: API call succeeded (model={model}, response='{text}')")
        else:
            print(f"OK: API call succeeded (model={model}) — model returned no "
                  f"visible content within the 50-token budget.  This is normal "
                  f"for reasoning models where reasoning tokens consume the "
                  f"budget; auth + credits + model id are all confirmed working.")
    except Exception as e:
        err = str(e)
        err_lower = err.lower()
        if "not a valid model id" in err_lower:
            if model.startswith("openrouter/"):
                native = model[len("openrouter/"):]
                print(f"FAIL: '{model}' has the LiteLLM-style 'openrouter/' prefix "
                      f"that OpenRouter's native API rejects.")
                print(f"  Fix: drop the prefix and pass '{native}' instead.")
            else:
                print(f"FAIL: '{model}' is not a valid OpenRouter model id. "
                      f"Check https://openrouter.ai/models for the canonical id.")
                print(f"  {e}")
        elif "401" in err or "no auth credentials" in err_lower or "invalid api key" in err_lower:
            print(f"FAIL: API key is invalid, revoked, or rejected by OpenRouter.\n  {e}")
        elif "402" in err or "insufficient_quota" in err_lower or "insufficient credits" in err_lower:
            print(f"FAIL: Account has insufficient credits (top-up required at openrouter.ai/credits).\n  {e}")
        elif "model_not_found" in err_lower or ("model" in err_lower and "not found" in err_lower):
            print(f"FAIL: Model '{model}' not accessible (check the model id, or that your tier permits it).\n  {e}")
        elif "rate_limit" in err_lower or "429" in err:
            print(f"WARN: Rate limited (key works but hitting limits).\n  {e}")
        elif "moderation" in err_lower or "403" in err:
            print(f"FAIL: Request blocked by OpenRouter or upstream provider moderation.\n  {e}")
        elif "no available providers" in err_lower or "503" in err:
            print(f"FAIL: No upstream provider currently serving '{model}' (transient — try again or pick another model).\n  {e}")
        else:
            print(f"FAIL: Unexpected error.\n  {e}")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
