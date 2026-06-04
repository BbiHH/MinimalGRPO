"""
src/tool_use_poc.py — Real Tool-Use Interactive Generation POC
==============================================================
Demonstrates a multi-turn generation loop where the model can call
a calculator tool via <calc>expression</calc> tags, the calculator
is actually executed, and the result is injected back into the
conversation for the model to continue from.

This is a standalone POC — it does NOT depend on the training pipeline
(rollout.py, reward.py, loss.py). Only tool.py is imported for the
calculator() and extract_calc_expressions() utilities.

Usage:
    python src/tool_use_poc.py

The script:
  1. Loads Qwen2.5-1.5B-Instruct (bfloat16, device_map="auto")
  2. Runs multi-turn generation on a hardcoded test prompt
  3. Prints each turn's output and the final conversation history
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
# Core: interactive multi-turn generation with tool interception
# ===================================================================

def _has_unclosed_calc(text: str) -> bool:
    """Return True if *text* contains an opening <calc> without a matching </calc>.

    This detects the case where max_new_tokens cut off the model mid-tag,
    so we can inform the user rather than silently treating it as "no tool use".
    """
    open_count = text.count("<calc>")
    close_count = text.count("</calc>")
    return open_count > close_count


@torch.no_grad()
def run_tool_use_loop(model, tokenizer, user_prompt: str) -> list[dict]:
    """Run multi-turn generation with real calculator tool interception.

    Loop outline:
      1. Build chat messages from the accumulated conversation history.
      2. Apply Qwen's chat template with add_generation_prompt=True, which
         appends "<|im_start|>assistant\\n" to prompt the model to respond.
      3. Generate up to max_new_tokens tokens (greedy, for determinism).
      4. Decode the **newly generated** tokens only.
      5. If <calc>expr</calc> tags are found:
           a. Extract each expression via extract_calc_expressions().
           b. Call calculator() to get the real result.
           c. Append the assistant's output as an "assistant" message.
           d. Append a "user" message containing <calc_result>result</calc_result>.
           e. Continue to the next turn.
      6. If no <calc> but <box> is found: the model gave a final answer — stop.
      7. If neither <calc> nor <box>: model answered without tools — stop.
      8. Stop after MAX_TURNS to prevent infinite loops.

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
    print(f"[POC] Multi-turn tool-use generation")
    print(f"[POC] User prompt: {user_prompt!r}")
    print(f"{'=' * 60}\n")

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn} / {MAX_TURNS} ---")

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
        # Step 2: Tokenize.
        # ---------------------------------------------------------------
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs.input_ids.shape[1]

        # ---------------------------------------------------------------
        # Step 3: Generate (greedy for reproducible POC).
        # ---------------------------------------------------------------
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS_PER_TURN,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # ---------------------------------------------------------------
        # Step 4: Slice off the input prompt, decode only what the model
        # generated this turn.
        # ---------------------------------------------------------------
        new_token_ids = outputs[0, input_len:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Show raw token count for observability.
        num_new_tokens = new_token_ids.shape[0]
        print(f"[POC] Generated {num_new_tokens} new tokens:")
        print(f"[POC]   {generated_text!r}")

        # ---------------------------------------------------------------
        # Step 5: Check for calculator expressions.
        # ---------------------------------------------------------------
        calc_expressions = extract_calc_expressions(generated_text)

        if calc_expressions:
            # --- Tool-use detected: execute the real calculator ---
            results: list[str] = []
            for expr in calc_expressions:
                result = calculator(expr)
                results.append(result)
                print(f"[POC]   => <calc>{expr}</calc> = {result}")

            # Append the assistant's partial / full response.
            messages.append({"role": "assistant", "content": generated_text})

            # Inject calculator result(s) as a tool-response message.
            # We use role="user" here because Qwen2.5's ChatML template
            # supports it natively; a production system would use
            # role="tool" with tool_call_id for proper OpenAI-style tracing.
            results_str = " ".join(
                f"<calc_result>{r}</calc_result>" for r in results
            )
            messages.append({"role": "user", "content": results_str})
            print(f"[POC]   Injected tool response: {results_str}")

            # Check for unclosed <calc> tags (generation cut off mid-tag).
            if _has_unclosed_calc(generated_text):
                print("[POC]   ⚠  Warning: unclosed <calc> tag detected "
                      "(generation may have been truncated)")

            # Check for both <calc> and <box> in the same turn.
            if "<box>" in generated_text:
                print("[POC]   (model also emitted <box> in this turn; "
                      "continuing anyway to show tool injection)")

        elif "<box>" in generated_text or "</box>" in generated_text:
            # --- Model provided a final answer (with or without tools) ---
            messages.append({"role": "assistant", "content": generated_text})
            print("[POC] Final answer detected (<box> found).  Stopping.")
            break

        else:
            # --- No tool use and no box — model answered without tools ---
            messages.append({"role": "assistant", "content": generated_text})
            if _has_unclosed_calc(generated_text):
                print("[POC] Unclosed <calc> tag — generation may have been "
                      "truncated by max_new_tokens.  Stopping.")
            else:
                print("[POC] No tool use or <box> detected.  Stopping.")
            break

        print()  # blank line between turns for readability

    else:
        # The for-loop completed without breaking — max turns exhausted.
        print(f"[POC] Max turns ({MAX_TURNS}) reached.  Stopping.")

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
