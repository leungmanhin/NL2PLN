"""
Probe DeepSeek model access via DSPy with reasoning effort.

Uses dspy.LM (the same interface this repo uses in production via simba.py /
nl2pln.py) so any working configuration here translates directly to those
scripts.  DSPy proxies to LiteLLM under the hood.

Conventions tested are based on documentation as of May 2026:

  DeepSeek direct API (api.deepseek.com, OpenAI-compatible):
    - `reasoning_effort` accepts: high, max, xhigh (xhigh maps to max).
      low and medium are aliases for high — DeepSeek does NOT actually
      support sub-high effort levels.
    - `thinking={"type": "enabled"}` toggles thinking mode on/off.
    - DeepSeek's docs recommend passing BOTH together via the OpenAI SDK:
        reasoning_effort="high"  +  extra_body={"thinking": {"type": "enabled"}}
      Via LiteLLM, top-level kwargs work: reasoning_effort="high",
      thinking={"type": "enabled"}.
    - Response: message.reasoning_content
    - Caveat: thinking mode disables temperature/top_p/penalties (silently).

  OpenRouter (openrouter.ai, unified reasoning API):
    - Uses extra_body={"reasoning": {"effort": "high"}} (NOT the
      reasoning_effort kwarg — that's OpenAI-style and OpenRouter has its
      own unified format).
    - Alternative: extra_body={"reasoning": {"max_tokens": N}} for direct
      reasoning-token budget control.
    - OpenRouter's effort_ratio: xhigh=0.95, high=0.8, medium=0.5,
      low=0.2, minimal=0.1.  Reasoning budget = max_tokens × effort_ratio,
      capped to [1024, 128000].
    - Response: message.reasoning  (NOTE: different field name than
      DeepSeek direct's message.reasoning_content)

Two providers tested:
  1. DeepSeek direct API   — set DEEPSEEK_API_KEY     (LiteLLM prefix: deepseek/)
  2. OpenRouter            — set OPENROUTER_API_KEY  (LiteLLM prefix: openrouter/)

For each provider, runs the same short reasoning-friendly prompt under
several parameter conventions and reports latency, token usage, and any
reasoning-token count the provider exposed.

Run:
    python misc/test_deepseek_dspy.py
    python misc/test_deepseek_dspy.py --model deepseek-v4-pro
    python misc/test_deepseek_dspy.py --skip-direct       # OpenRouter only
    python misc/test_deepseek_dspy.py --skip-openrouter   # direct only
    python misc/test_deepseek_dspy.py --effort max        # max-reasoning probe
    python misc/test_deepseek_dspy.py --verbose           # show full LM history entries

Sources (May 2026):
  https://api-docs.deepseek.com/guides/thinking_mode
  https://docs.litellm.ai/docs/providers/deepseek
  https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
"""
import argparse
import os
import sys
import time
from pprint import pprint


# A small prompt that benefits from reasoning: the answer requires combining
# two facts and rejecting an undistributed-middle fallacy.  A non-reasoning
# model often answers "yes"; a reasoning model sees the gap and answers "no".
TEST_PROMPT = (
    "All squares are rectangles. Some rectangles are blue. "
    "Does it follow that all squares are blue? "
    "Answer 'yes' or 'no' and give a one-sentence justification."
)


def _get(obj, key, default=None):
    """Read `key` from a dict OR attribute from a Pydantic-style object.

    LiteLLM's response.usage and usage.completion_tokens_details are
    typed wrappers that don't support .get() — only attribute access.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_call(label, lm_factory, verbose=False):
    """Build an LM via lm_factory() and call it with TEST_PROMPT.

    Two-phase try blocks:
      (1) the LM call itself — failure here means the config is wrong
      (2) post-call usage extraction — failure here is non-fatal,
          shouldn't invalidate ok status (we still have the response).
    """
    print(f"\n=== {label} ===")
    result = {"label": label, "ok": False}
    t0 = time.time()

    # Phase 1: build LM and make the call.
    try:
        lm = lm_factory()
        response = lm(TEST_PROMPT)
    except Exception as e:
        result["latency_s"] = round(time.time() - t0, 2)
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  ok=False  latency={result['latency_s']}s")
        err_msg = result["error"]
        print(f"  error: {err_msg[:600]}{'...' if len(err_msg) > 600 else ''}")
        return result

    # Phase 1 succeeded: lock in the result before any further work.
    result["latency_s"] = round(time.time() - t0, 2)
    result["ok"] = True
    # response[0] is normally a string but DSPy 3.1.x returns a dict for
    # reasoning models that expose both text and reasoning fields.
    raw_first = response[0] if response else ""
    if isinstance(raw_first, dict):
        text = (
            raw_first.get("text")
            or raw_first.get("content")
            or str(raw_first)
        )
    else:
        text = str(raw_first)
    result["response"] = text
    print(f"  ok=True  latency={result['latency_s']}s")

    # Phase 2: extract usage info.  Failures here are warnings, not errors.
    try:
        if lm.history:
            entry = lm.history[-1]
            usage = _get(entry, "usage")
            result["total_tokens"] = _get(usage, "total_tokens")
            details = _get(usage, "completion_tokens_details")
            result["reasoning_tokens"] = _get(details, "reasoning_tokens")

            # Reasoning content is in different fields by provider:
            #   DeepSeek direct -> message.reasoning_content
            #   OpenRouter      -> message.reasoning
            raw_response = _get(entry, "response")
            try:
                msg = raw_response.choices[0].message
                rc = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                )
                if rc:
                    result["reasoning_content_chars"] = len(rc)
                    result["reasoning_excerpt"] = rc[:200]
            except (AttributeError, IndexError, TypeError):
                pass

            if verbose:
                print("  --- full history entry ---")
                pprint({k: v for k, v in entry.items() if k != "response"})
                print("  --- end ---")
    except Exception as ex:
        print(f"  (warning: usage extraction failed: {type(ex).__name__}: {ex})")

    # Print what we extracted (or what we didn't).
    if result.get("reasoning_tokens"):
        print(f"  reasoning_tokens={result['reasoning_tokens']}, "
              f"total_tokens={result.get('total_tokens')}  "
              f"<-- reasoning was active")
    elif result.get("total_tokens"):
        print(f"  total_tokens={result['total_tokens']}  "
              f"(no reasoning_tokens reported by provider)")
    else:
        print("  (no usage info exposed)")
    if result.get("reasoning_content_chars"):
        excerpt = result["reasoning_excerpt"].replace("\n", " ")
        print(f"  reasoning_content: {result['reasoning_content_chars']} chars; "
              f"excerpt: {excerpt}{'...' if result['reasoning_content_chars'] > 200 else ''}")
    snippet = text[:300].replace("\n", " ")
    print(f"  response: {snippet}{'...' if len(text) > 300 else ''}")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe DeepSeek model access via DSPy/LiteLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", default="deepseek-v4-pro",
        help="DeepSeek model name without provider prefix.  Examples: "
             "deepseek-v4-pro, deepseek-reasoner, deepseek-chat.  "
             "(default: deepseek-v4-pro)",
    )
    p.add_argument(
        "--openrouter-model", default=None,
        help="Override the model name when going through OpenRouter, since "
             "OpenRouter sometimes uses different ids than DeepSeek's direct "
             "API.  Defaults to 'deepseek/<--model>'.",
    )
    p.add_argument(
        "--skip-direct", action="store_true",
        help="Skip the DeepSeek direct API tests",
    )
    p.add_argument(
        "--skip-openrouter", action="store_true",
        help="Skip the OpenRouter tests",
    )
    p.add_argument(
        "--effort", default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort to request (default: high).  Note for DeepSeek "
             "direct: 'low' and 'medium' are silently mapped to 'high'; only "
             "'high' (default) and 'max' (or 'xhigh' alias) are real distinct "
             "levels.  For OpenRouter: all five levels map to different "
             "effort_ratios and produce different budgets.",
    )
    p.add_argument(
        "--max-reasoning-tokens", type=int, default=8000,
        help="Cap on reasoning tokens for the max-tokens variant (default: 8000)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Cap on output tokens (default: 1024).  Set high enough that "
             "reasoning models can finish; low budgets cause silent truncation.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print the full lm.history[-1] entry for each successful call",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Import here so --help works without dspy installed (matches the rest
    # of this repo's pattern).
    import dspy
    print(f"DSPy version: {dspy.__version__}")
    print(f"Test prompt: {TEST_PROMPT}")
    print(f"Effort requested: {args.effort}")
    print(f"max_tokens (output cap): {args.max_tokens}")

    # Disable DSPy's response caching for these probes; we want fresh calls
    # for every config so identical-prompt results don't hide differences.
    base_kwargs = {"max_tokens": args.max_tokens, "cache": False}

    results = []

    # ---------- DeepSeek direct API ----------
    if not args.skip_direct:
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("\nDEEPSEEK_API_KEY not set — skipping direct DeepSeek tests.\n"
                  "(Set it with: export DEEPSEEK_API_KEY=sk-...)")
        else:
            model_id = f"deepseek/{args.model}"
            print(f"\nDeepSeek direct API tests against {model_id}:")

            # (a) reasoning_effort only (LiteLLM-supported per docs)
            results.append(_safe_call(
                f"direct: {model_id} + reasoning_effort='{args.effort}'",
                lambda: dspy.LM(model_id, reasoning_effort=args.effort, **base_kwargs),
                args.verbose,
            ))

            # (b) thinking only (LiteLLM-supported per docs)
            results.append(_safe_call(
                f"direct: {model_id} + thinking={{'type':'enabled'}}",
                lambda: dspy.LM(
                    model_id,
                    thinking={"type": "enabled"},
                    **base_kwargs,
                ),
                args.verbose,
            ))

            # (c) DeepSeek's official combined pattern (their OpenAI-SDK
            # docs recommend BOTH together; both should be redundant via
            # LiteLLM, but we test it explicitly to verify it doesn't error).
            results.append(_safe_call(
                f"direct: {model_id} + reasoning_effort + thinking (combined)",
                lambda: dspy.LM(
                    model_id,
                    reasoning_effort=args.effort,
                    thinking={"type": "enabled"},
                    **base_kwargs,
                ),
                args.verbose,
            ))

            # (d) Control: no reasoning param (verifies base call works
            # and confirms baseline latency for comparison)
            results.append(_safe_call(
                f"direct: {model_id} + no reasoning param (control)",
                lambda: dspy.LM(model_id, **base_kwargs),
                args.verbose,
            ))

    # ---------- OpenRouter ----------
    if not args.skip_openrouter:
        if not os.getenv("OPENROUTER_API_KEY"):
            print("\nOPENROUTER_API_KEY not set — skipping OpenRouter tests.\n"
                  "(Set it with: export OPENROUTER_API_KEY=sk-or-...)")
        else:
            model_part = args.openrouter_model or f"deepseek/{args.model}"
            model_id = f"openrouter/{model_part}"
            print(f"\nOpenRouter tests against {model_id}:")

            # (a) OpenRouter's unified `reasoning.effort` (documented format).
            # This is the recommended path per OpenRouter docs.
            results.append(_safe_call(
                f"openrouter: {model_id} + extra_body=reasoning.effort='{args.effort}'",
                lambda: dspy.LM(
                    model_id,
                    extra_body={"reasoning": {"effort": args.effort}},
                    **base_kwargs,
                ),
                args.verbose,
            ))

            # (b) OpenRouter's `reasoning.max_tokens` (direct token budget).
            # Useful when you want a precise reasoning-token cap rather than
            # a coarse effort level.
            results.append(_safe_call(
                f"openrouter: {model_id} + extra_body=reasoning.max_tokens={args.max_reasoning_tokens}",
                lambda: dspy.LM(
                    model_id,
                    extra_body={"reasoning": {"max_tokens": args.max_reasoning_tokens}},
                    **base_kwargs,
                ),
                args.verbose,
            ))

            # (c) Top-level reasoning_effort (long-shot — OpenRouter prefers
            # the unified extra_body format, but LiteLLM may translate the
            # OpenAI-style kwarg.  Worth probing in case it works.)
            results.append(_safe_call(
                f"openrouter: {model_id} + reasoning_effort='{args.effort}' (long-shot)",
                lambda: dspy.LM(model_id, reasoning_effort=args.effort, **base_kwargs),
                args.verbose,
            ))

            # (d) Control: no reasoning param
            results.append(_safe_call(
                f"openrouter: {model_id} + no reasoning param (control)",
                lambda: dspy.LM(model_id, **base_kwargs),
                args.verbose,
            ))

    # ---------- Summary ----------
    if not results:
        print("\nNo tests run.  Set DEEPSEEK_API_KEY and/or OPENROUTER_API_KEY.")
        sys.exit(1)

    print("\n" + "=" * 110)
    print("SUMMARY (reasoning_tokens > 0 OR reasoning_content present confirms reasoning was active)")
    print("=" * 110)
    print(f"{'config':<78} {'ok':<4} {'latency':<8} {'reason_tok':<10} {'rc_chars':<8}")
    print("-" * 110)
    for r in results:
        cfg = r["label"][:78]
        ok = "yes" if r["ok"] else "NO"
        lat = f"{r.get('latency_s', '?')}s"
        rt = r.get("reasoning_tokens")
        rt_str = str(rt) if rt is not None else "—"
        rc = r.get("reasoning_content_chars")
        rc_str = str(rc) if rc is not None else "—"
        print(f"{cfg:<78} {ok:<4} {lat:<8} {rt_str:<10} {rc_str:<8}")

    # A config "shows reasoning" if either token count or content field is present.
    working_with_reasoning = [
        r for r in results
        if r["ok"] and (
            (r.get("reasoning_tokens") or 0) > 0
            or r.get("reasoning_content_chars")
        )
    ]
    if working_with_reasoning:
        print("\nWorking configs with reasoning active:")
        for r in working_with_reasoning:
            print(f"  - {r['label']}  ({r['reasoning_tokens']} reasoning tokens)")
        print("\nUse the corresponding kwargs in dspy.LM(...) for production calls.")
    else:
        ok_results = [r for r in results if r["ok"]]
        if ok_results:
            print("\nNo config showed reasoning_tokens > 0.")
            print("This may mean: (a) the model isn't a reasoning model under any "
                  "config we tried, (b) the provider doesn't expose reasoning_tokens "
                  "in usage, or (c) all our parameter conventions are silently "
                  "ignored.  Check latency: a reasoning-active call usually takes "
                  "noticeably longer than a non-reasoning one.")
        else:
            print("\nNo configs succeeded.  Common causes: API key not set or invalid, "
                  "model id wrong (try a known model like 'deepseek-chat'), or "
                  "insufficient credits.")


if __name__ == "__main__":
    main()
