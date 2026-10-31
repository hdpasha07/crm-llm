import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Define base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "langgraph_agents_train.jsonl"
ENV_FILE = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_FILE)

# System prompt defining data engineer role and LangGraph agent schemas
SYSTEM_PROMPT = """You are a senior data engineer specializing in synthetic dataset generation for fine-tuning multi-agent LLM systems using Unsloth.
Your task is to generate realistic, diverse customer complaints and convert each into training examples for four specialized LangGraph agents.

For each request, you must generate exactly 5 distinct customer complaints across diverse domains (e.g., e-commerce, banking, logistics, SaaS, telecommunications).
For EACH of the 5 complaints, produce 4 training examples corresponding to four LangGraph agent roles:

1. Categorizer Agent:
   - Role: Classifies the customer complaint into an appropriate category.
   - System: "You are a customer complaint categorizer agent. Classify the customer complaint into an appropriate category. Output strict JSON with key 'category'."
   - Input: The raw customer complaint text.
   - Output: An escaped JSON string with key 'category'. Example: "{\\"category\\": \\"Billing Issue\\"}"

2. Sentiment Agent:
   - Role: Analyzes customer sentiment and urgency priority.
   - System: "You are a customer sentiment and priority analysis agent. Analyze the sentiment and priority of the complaint. Output strict JSON with keys 'sentiment' and 'priority'."
   - Input: The raw customer complaint text.
   - Output: An escaped JSON string with keys 'sentiment' and 'priority'. Example: "{\\"sentiment\\": \\"Negative\\", \\\"priority\\\": \\\"High\\"}"

3. Policy Agent:
   - Role: Retrieves and summarizes applicable company policy based on category and dummy RAG context.
   - System: "You are a CRM policy agent. Given the complaint category and relevant policy context, summarize the applicable policy for resolving this issue. Output strict JSON with key 'policy_summary'."
   - Input: The complaint category along with realistic dummy RAG context (e.g., "Category: Billing Issue\\nContext: Under Policy Section 4.1, unauthorized or duplicate recurring charges are eligible for immediate full refund within 30 days of the transaction.").
   - Output: An escaped JSON string with key 'policy_summary'. Example: "{\\"policy_summary\\": \\"Eligible for immediate refund under Section 4.1 if reported within 30 days.\\"}"

4. Response Agent:
   - Role: Drafts a professional customer resolution email based on the complaint and policy.
   - System: "You are a customer response agent. Given the customer complaint and the policy summary, draft a professional email response addressing the customer's issue. Output strict JSON with key 'email_response'."
   - Input: The customer complaint text and policy summary (e.g., "Complaint: <complaint text>\\nPolicy: <policy summary>").
   - Output: An escaped JSON string with key 'email_response'. Example: "{\\"email_response\\": \\"Dear Customer,\\\\n\\\\nThank you for reaching out. We apologize for the issue...\\"}"

OUTPUT FORMAT REQUIREMENTS:
Return a raw JSON array containing exactly 20 objects (5 complaints x 4 agent roles).
Each object MUST strictly have the following keys:
- "system": The agent's system prompt string.
- "input": The input string provided to the agent.
- "output": The target response as an ESCAPED JSON string (do NOT provide a nested JSON object for output; it MUST be a valid escaped JSON string).

Do not include any introductory remarks, markdown code block backticks (e.g. ```json), or trailing text. Return ONLY the raw JSON array.
"""


def clean_markdown_fences(text: str) -> str:
    """Removes markdown code fences (```json ... ```) from the Gemini response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    load_dotenv(dotenv_path=ENV_FILE)
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print(
            "ERROR: GEMINI_API_KEY is missing or contains the default placeholder in .env."
        )
        print("Please set a valid GEMINI_API_KEY in:", ENV_FILE)
        sys.exit(1)

    # Configure Gemini SDK and initialize model
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(
        model_name="Gemini 2.5 Flash Lite",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    TOTAL_BATCHES = 10
    total_saved_examples = 0
    max_retries_per_batch = 3

    print("==========================================================")
    print("Starting Synthetic Training Data Generation Pipeline")
    print(f"Target: {TOTAL_BATCHES} batches (20 examples/batch = 200 total)")
    print(f"Output File: {OUTPUT_FILE}")
    print("==========================================================\n")

    for batch_num in range(1, TOTAL_BATCHES + 1):
        print(f"--- [Batch {batch_num}/{TOTAL_BATCHES}] Generating 5 complaints (20 agent examples)... ---")
        success = False

        for attempt in range(1, max_retries_per_batch + 1):
            try:
                user_prompt = (
                    f"Generate batch #{batch_num} with 5 brand-new, unique customer complaints "
                    f"and their corresponding 20 LangGraph agent training examples (4 agents per complaint). "
                    f"Ensure diverse industries, realistic scenarios, and valid escaped JSON strings for 'output'."
                )

                response = model.generate_content(user_prompt)
                raw_text = response.text or ""
                cleaned_text = clean_markdown_fences(raw_text)

                data = json.loads(cleaned_text)

                if not isinstance(data, list):
                    raise ValueError(f"Expected JSON array, received: {type(data).__name__}")

                batch_count = 0
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        
                        # Validate required keys
                        if "system" not in item or "input" not in item or "output" not in item:
                            continue

                        # Ensure output is serialized as an escaped JSON string
                        if isinstance(item["output"], (dict, list)):
                            item["output"] = json.dumps(item["output"])
                        elif not isinstance(item["output"], str):
                            item["output"] = str(item["output"])

                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        batch_count += 1

                total_saved_examples += batch_count
                print(f"[OK] Batch {batch_num} successful: Saved {batch_count} examples. Total saved so far: {total_saved_examples}")
                success = True
                break

            except json.JSONDecodeError as json_err:
                print(f"  [Attempt {attempt}/{max_retries_per_batch}] JSON parsing error: {json_err}")
            except Exception as e:
                print(f"  [Attempt {attempt}/{max_retries_per_batch}] Generation error: {e}")

            if attempt < max_retries_per_batch:
                wait_time = attempt * 3
                print(f"  Retrying batch {batch_num} in {wait_time}s...")
                time.sleep(wait_time)

        if not success:
            print(f"[FAIL] Failed to generate Batch {batch_num} after {max_retries_per_batch} attempts. Continuing to next batch...")

        # Modest rate-limit delay between batches
        time.sleep(1)

    print("\n==========================================================")
    print(f"Pipeline Completed! Total examples written: {total_saved_examples}")
    print(f"Dataset location: {OUTPUT_FILE}")
    print("==========================================================")


if __name__ == "__main__":
    main()

