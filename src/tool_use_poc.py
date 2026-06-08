"""
src/tool_use_poc.py — Real Token-by-Token Tool-Use Generation POC
==================================================================
Demonstrates true autoregressive generation with real-time interception of
calculator tool calls.  Instead of calling model.generate() (batch generation
that can overshoot), this POC performs manual token-by-token generation with
KV-cache, decoding after every new token to check whether ``</calc>`` has
appeared.  When it does, generation stops immediately, the calculator is
executed, and the result is injected back into the conversation for the model
to continue from.

This is a standalone POC — it does NOT depend on the training pipeline
(rollout.py, reward.py, loss.py).  Only tool.py is imported for the
calculator() and extract_calc_expressions() utilities.

Usage::

    python src/tool_use_poc.py

The script:
  1. Loads Qwen2.5-1.5B-Instruct (bfloat16, device_map="auto")
  2. Runs multi-turn token-by-token generation on a hardcoded test prompt
  3. Prints streaming output and the final conversation history
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure the src/ directory is importable so we can reach tool.py
# regardless of how the script is invoked (python src/tool_use_poc.py
# or python -m src.tool_use_poc).  When run as a script, Python adds the
# script's directory to sys.path[0]; when run as a module the current
# working directory is used instead.  This guard covers both cases.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from tool import extract_calc_expressions, calculator


# ===================================================================
# Configuration
# ===================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
MAX_TURNS = 5
MAX_NEW_TOKENS_PER_TURN = 50
SEED = 42

# Mirrors rollout.py's TOOL_SYSTEM_PROMPT — replicated here to keep this
# script self-contained (per the constraint: no import from rollout.py).
TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator. "
    "When you need to perform arithmetic, you MUST use the calculator "
    "by writing <calc>expression</calc>. Then provide the final answer."
)

# The hardcoded test prompt for the POC.
# (This exact prompt appears in data/tool_calling_prompts.jsonl line 4,
# with answer 5137 — so we can verify correctness.)
TEST_USER_PROMPT = "Compute the sum: 1545 + 3592."


# ===================================================================
# Model loading
# ===================================================================

def load_model_and_tokenizer(model_name: str, dtype: torch.dtype):
    """Load the Qwen2.5-Instruct model and tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
        dtype:      Torch dtype for model weights (bfloat16 recommended).

    Returns:
        (model, tokenizer) tuple.  model is in eval mode on the GPU(s)
        determined by device_map="auto".
    """
    print(f"[POC] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[POC]   pad_token set to eos_token")

    print(f"[POC] Loading model: {model_name}  (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    print(f"[POC]   Model loaded: {total_params / 1e9:.2f}B params on {device}")

    return model, tokenizer


# ===================================================================
# Helpers
# ===================================================================

def _has_unclosed_calc(text: str) -> bool:
    """Return True if *text* contains an opening <calc> without a matching </calc>.

    This detects the case where max_new_tokens cut off the model mid-tag,
    so we can inform the user rather than silently treating it as "no tool use".
    """
    return text.count("<calc>") > text.count("</calc>")


# ===================================================================
# Core: token-by-token generation with real-time tool interception
# ===================================================================

@torch.no_grad()
def run_tool_use_loop(model, tokenizer, user_prompt: str) -> list[dict]:
    """Run multi-turn token-by-token generation with real calculator interception.

    Unlike the previous batch-generation approach (model.generate()), this
    function performs true autoregressive token-by-token generation using
    model.forward() with KV-cache (past_key_values).  After each new token
    the accumulated text is decoded and checked for the ``</calc>`` stop
    marker.  When found, generation stops **immediately** — the calculator is
    executed and its result is injected back into the conversation before the
    model can continue.

    Loop outline for each turn:
      1. Build chat messages from the accumulated conversation history.
      2. Apply Qwen's chat template with ``add_generation_prompt=True``.
      3. First forward pass on the full input to get ``past_key_values``
         and the logits for the first new token.
      4. **Token-by-token loop**:
           a. Greedy-pick the next token (argmax).
           b. If EOS → stop this turn.
           c. Append token to the generated list.
           d. Decode ALL generated tokens so far to a text string.
           e. Check: does text contain ``</calc>``?
              YES → stop generating, extract expression(s), call
                    calculator(), inject result as a new message,
                    go to next turn.
           f. Check: does text contain ``<box>`` (and no unclosed ``<calc>``)?
              YES → generation complete, exit outer loop.
           g. Check: reached ``max_new_tokens``?
              YES → stop this turn.
           h. Forward pass on the single new token with ``past_key_values``
              to get logits for the next position.
      5. Post-generation: dispatch based on the stop reason.

    Args:
        model:       HuggingFace CausalLM in eval mode.
        tokenizer:   HuggingFace tokenizer (pad_token already configured).
        user_prompt: The user's question string.

    Returns:
        messages: The full conversation history as a list of message dicts
                  with "role" and "content" keys.
    """
    # --- Initialise conversation ---
    messages: list[dict] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"\n{'=' * 60}")
    print(f"[POC] Token-by-token tool-use generation")
    print(f"[POC] User prompt: {user_prompt!r}")
    print(f"[POC] Max turns: {MAX_TURNS}  |  Max new tokens/turn: {MAX_NEW_TOKENS_PER_TURN}")
    print(f"{'=' * 60}")

    eos_token_id = tokenizer.eos_token_id

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- Turn {turn} / {MAX_TURNS} ---")

        # ---------------------------------------------------------------
        # Step 1: Build the full prompt from the conversation so far.
        # add_generation_prompt=True tells the tokenizer to finish the
        # template with "<|im_start|>assistant\n" so the model knows it
        # should now produce the assistant's reply.
        # ---------------------------------------------------------------
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ---------------------------------------------------------------
        # Step 2: Tokenize the full conversation.
        # ---------------------------------------------------------------
        enc = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_ids = enc.input_ids  # shape: (1, input_len)
        input_len = input_ids.shape[1]

        # ---------------------------------------------------------------
        # Step 3: First forward pass on the full input to establish
        # the KV-cache and get logits for the first generated token.
        # No attention_mask is passed — for a single non-padded sequence
        # the model's internal causal mask (is_causal=True) is sufficient.
        # ---------------------------------------------------------------
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        # Logits at the last input position → first token to generate
        logits_last = out.logits[0, -1, :]  # shape: (vocab_size,)
        next_token_scalar = torch.argmax(logits_last)  # scalar (0-dim) tensor

        # ---------------------------------------------------------------
        # Step 4: Token-by-token generation loop.
        # ---------------------------------------------------------------
        generated_ids: list[int] = []
        generated_text: str = ""
        stop_reason: str = ""  # "eos" | "calc" | "box" | "max_tokens"

        # Streaming header for this turn
        print("[assistant] ", end="", flush=True)

        for _step in range(1, MAX_NEW_TOKENS_PER_TURN + 1):
            token_id = next_token_scalar.item()

            # -- 4a. Check EOS ------------------------------------------
            if token_id == eos_token_id:
                stop_reason = "eos"
                break

            # -- 4b. Append token & decode all generated text -----------
            generated_ids.append(token_id)
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            )

            # -- Streaming print: show the new token's text immediately -
            new_piece = tokenizer.decode([token_id], skip_special_tokens=True)
            print(new_piece, end="", flush=True)

            # -- 4c. Check for </calc> (tool-call closing tag) ----------
            if "</calc>" in generated_text:
                stop_reason = "calc"
                break

            # -- 4d. Check for <box> (final-answer marker) --------------
            # Only treat <box> as a stop signal when there is no
            # unclosed <calc> — otherwise we may be inside a malformed
            # expression and should keep generating.
            if "<box>" in generated_text and not _has_unclosed_calc(generated_text):
                stop_reason = "box"
                break

            # -- 4e. Forward pass for the next token (KV-cached) --------
            # Feed only the single new token; past_key_values avoids
            # recomputing the entire prefix.
            next_input = next_token_scalar.view(1, 1)  # shape: (1, 1)
            out = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            # Logits at position 0 (the only position in this step)
            logits_last = out.logits[0, -1, :]
            next_token_scalar = torch.argmax(logits_last)

        # ---------------------------------------------------------------
        # End of token loop — finalise the streaming line.
        # ---------------------------------------------------------------
        print()  # newline after streaming output

        num_generated = len(generated_ids)
        effective_stop = stop_reason if stop_reason else "max_tokens"
        print(f"[POC] Stop reason: {effective_stop}  |  "
              f"Tokens generated this turn: {num_generated}")
        print(f"[POC] Generated text: {generated_text!r}")

        # ---------------------------------------------------------------
        # Step 5: Post-generation dispatch.
        # ---------------------------------------------------------------

        # --- Case A: Calculator tool call detected ---------------------
        if stop_reason == "calc" or "</calc>" in generated_text:
            calc_expressions = extract_calc_expressions(generated_text)

            if calc_expressions:
                results: list[str] = []
                for expr in calc_expressions:
                    result = calculator(expr)
                    results.append(result)
                    print(f"[POC]   => <calc>{expr}</calc> = {result}")

                # Append the assistant's partial / full response.
                messages.append({"role": "assistant", "content": generated_text})

                # Inject calculator result(s) as a tool-response message.
                # We use role="user" because Qwen2.5's ChatML template
                # supports it natively; a production system would use
                # role="tool" with tool_call_id.
                results_str = " ".join(
                    f"<calc_result>{r}</calc_result>" for r in results
                )
                messages.append({"role": "user", "content": results_str})
                print(f"[POC]   Injected tool response: {results_str}")

                # Warn if generation was cut off mid-tag.
                if _has_unclosed_calc(generated_text):
                    print("[POC]   ⚠  Warning: unclosed <calc> tag "
                          "(generation truncated mid-expression)")

                # If the model also emitted <box> in the same burst
                # (rare but possible with token-by-token), note it.
                if "<box>" in generated_text:
                    print("[POC]   (model also emitted <box> in this turn; "
                          "continuing to next turn)")
            else:
                # extract_calc_expressions returned empty — this can happen
                # if </calc> appears in a non-expression context (e.g.,
                # in a code block).  Treat as a regular response.
                print("[POC]   </calc> found but no valid expression extracted — "
                      "treating as plain response")
                messages.append({"role": "assistant", "content": generated_text})
                break

        # --- Case B: Final answer marker detected ----------------------
        elif stop_reason == "box" or "<box>" in generated_text:
            messages.append({"role": "assistant", "content": generated_text})
            print("[POC] Final answer detected (<box> found).  Stopping.")
            break

        # --- Case C: No tool use, no box — plain response -------------
        else:
            messages.append({"role": "assistant", "content": generated_text})

            if stop_reason == "eos":
                print("[POC] EOS token generated.  Stopping.")
            elif _has_unclosed_calc(generated_text):
                print("[POC] ⚠  Warning: unclosed <calc> tag — "
                      "generation truncated by max_tokens.  Stopping.")
            else:
                print("[POC] No tool use or <box> detected.  Stopping.")
            break

    else:
        # The for-loop completed without breaking — max turns exhausted.
        print(f"\n[POC] Max turns ({MAX_TURNS}) reached.  Stopping.")

    return messages


# ===================================================================
# Display helpers
# ===================================================================

def print_conversation_history(messages: list[dict]) -> None:
    """Pretty-print the full conversation history with role labels."""
    print(f"\n{'=' * 60}")
    print("[POC] Full Conversation History")
    print(f"{'=' * 60}")

    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        # For very long messages, truncate for readability while
        # showing that content was clipped.
        max_display = 300
        display_content = content if len(content) <= max_display \
            else content[:max_display] + "…"

        print(f"\n[{i}] {role}:")
        # Indent multi-line content for clarity.
        for line in display_content.split("\n"):
            print(f"    {line}")

    print(f"\n{'=' * 60}")
    print(f"[POC] Total messages: {len(messages)}")
    print(f"{'=' * 60}")


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    """Entry point: load model, run tool-use loop, display results."""

    # --- Reproducibility ---
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, DTYPE)

    # --- Run interactive tool-use loop ---
    messages = run_tool_use_loop(model, tokenizer, TEST_USER_PROMPT)

    # --- Display the full conversation ---
    print_conversation_history(messages)

    # --- Clean up ---
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[POC] Done.")


if __name__ == "__main__":
    main()
