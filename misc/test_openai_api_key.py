"""
Quick diagnostic for OpenAI API key issues.

Usage:
    python tests/test_openai_api_key.py [MODEL]

Checks, in order:
    1. OPENAI_API_KEY is set in the environment
    2. The key is accepted by the API (not invalid/revoked)
    3. The account has sufficient quota
"""
import os
import sys


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4-mini"

    # 1. Check env var
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("FAIL: OPENAI_API_KEY is not set.")
        print("  Fix: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    masked = api_key[:7] + "..." + api_key[-4:]
    print(f"OK: OPENAI_API_KEY is set ({masked})")

    # 2-3. Try a minimal API call
    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: 'openai' package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_completion_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        print(f"OK: API call succeeded (model={model}, response='{text}')")
    except Exception as e:
        err = str(e)
        if "Incorrect API key" in err or "invalid_api_key" in err:
            print(f"FAIL: API key is invalid or revoked.\n  {e}")
        elif "insufficient_quota" in err or "exceeded your current quota" in err:
            print(f"FAIL: Account has insufficient quota (billing issue).\n  {e}")
        elif "account_deactivated" in err or "account has been deactivated" in err.lower():
            print(f"FAIL: Account is deactivated/banned.\n  {e}")
        elif "model_not_found" in err or "does not exist" in err:
            print(f"FAIL: Model '{model}' not found. Check the model name.\n  {e}")
        elif "rate_limit" in err.lower():
            print(f"WARN: Rate limited (key works but hitting limits).\n  {e}")
        else:
            print(f"FAIL: Unexpected error.\n  {e}")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
