import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "Qwen2.5-7B-Instruct",
)
DEFAULT_MODEL_PATH = os.path.expanduser(
    os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)
)

DEFAULT_SYSTEM_PROMPT = """You are an expert pathology planner.

The available tissue regions are:

1. Tumor
2. Benign epithelium
3. Stroma
4. Necrosis
5. Lymphocytes
6. Blood vessel
7. Nerve
8. Adipose tissue
9. Smooth muscle
10. Inflammation
11. Hemorrhage
12. Background
13. Other

Your task is NOT to answer the pathology question.

Instead, decide:

1. Which tissue regions should be examined?
2. Which magnification should be used?
3. Do NOT attempt to answer the question.

Output ONLY:

Region(s):
Magnification:
Reason:

Use exactly these three field names in plain text. Do not use Markdown, bullet
points, italics, bold text, or alternative labels such as "Tissue(s)".
Use only concise pathology terms. For Magnification, use common microscope
magnifications such as 5x, 10x, 20x, or 40x. Do not invent extremely large
numbers. Keep the answer brief. Do not write tool calls, XML tags, chat
template tokens, or simulated user messages.

The Reason must explain why the selected tissue region(s) and magnification are
appropriate for planning how to answer the question. Do not state the final
diagnosis, grade, Gleason score, recurrence risk, or other pathology conclusion.

The Region(s) field must include every tissue region that is essential to the
Reason. If the Reason mentions tumor cells, malignant cells, carcinoma, or
neoplasm, include Tumor in Region(s). If the Reason mentions nerves, include
Nerve. Do not list only the surrounding tissue when the diagnostic evidence
depends on tumor interacting with another structure. Do not include Background
unless the question or Reason explicitly concerns background, artifact, or
non-tissue regions. For lymphovascular invasion, examine Tumor and Blood vessel;
do not include Lymphocytes unless the question specifically asks about
lymphocytic inflammation."""


def configure_utf8_output():
    """Avoid garbled Chinese output in terminals with a non-UTF-8 default."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chat with a local Qwen2.5-Instruct model."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--keep_history",
        action="store_true",
        help="Keep previous turns in context. Off by default for stable planning.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = None
    return model, tokenizer


def get_eos_token_ids(tokenizer):
    eos_token_ids = []
    for token in (tokenizer.eos_token, "<|im_end|>"):
        if not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            eos_token_ids.append(token_id)
    return sorted(set(eos_token_ids))


def clean_response(response, question=""):
    stop_texts = (
        "<|im_end|>",
        "<|im_start|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "\nuser\n",
        "\nUser:",
        "\nYou:",
        "\nassistant\n",
    )
    for stop_text in stop_texts:
        if stop_text in response:
            response = response.split(stop_text, 1)[0]
    response = normalize_format(response.strip())
    response = make_regions_consistent(response)
    response = enforce_three_field_format(response)
    return guard_unsupported_conclusions(response, question)


def clean_user_input(user_input):
    cleaned_lines = []
    for line in user_input.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"(?i)^qwen\s*:", line):
            continue
        while re.match(r"(?i)^(you|user)\s*:", line):
            line = re.sub(r"(?i)^(you|user)\s*:\s*", "", line, count=1).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_format(response):
    response = response.replace("*", "")
    response = normalize_magnification_text(response)
    response = re.sub(r"(?im)^\s*region\(s\)\s*:", "Region(s):", response)
    response = re.sub(r"(?im)^\s*tissue\(s\)\s*[-:]\s*", "Region(s): ", response)
    response = re.sub(r"(?im)^\s*magnification\s*:", "Magnification:", response)
    response = re.sub(r"(?im)^\s*reason\s*:", "Reason:", response)
    return response.strip()


def normalize_magnification_text(text):
    def replace_match(match):
        value = int(match.group(1))
        allowed = [5, 10, 20, 40]
        nearest = min(allowed, key=lambda item: abs(item - value))
        return f"{nearest}x"

    return re.sub(r"\b(\d{1,3})\s*[xX×]\b", replace_match, text)


def make_regions_consistent(response):
    region_match = re.search(
        r"(Region\(s\):\s*)(.*?)(\n\s*Magnification:)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reason_match = re.search(
        r"Reason:\s*(.*)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not region_match or not reason_match:
        return response

    prefix, regions_text, suffix = region_match.groups()
    reason = reason_match.group(1).lower()
    regions_lower = regions_text.lower()
    required_regions = []

    if (
        any(term in reason for term in ("tumor", "malignant", "carcinoma", "neoplasm"))
        and "tumor" not in regions_lower
    ):
        required_regions.append("Tumor")
    if "nerve" in reason and "nerve" not in regions_lower:
        required_regions.append("Nerve")

    if not required_regions:
        return response

    cleaned_regions = regions_text.strip()
    separator = ", " if cleaned_regions and not cleaned_regions.endswith((",", ";")) else " "
    new_regions = f"{cleaned_regions}{separator}{', '.join(required_regions)}"
    return (
        response[:region_match.start()]
        + prefix
        + new_regions
        + suffix
        + response[region_match.end():]
    )


def enforce_three_field_format(response):
    text = response.strip()
    fields = parse_response_fields(text)

    regions = fields.get("regions", "")
    magnification = fields.get("magnification", "")
    reason = fields.get("reason", "")

    if not reason:
        reason = remove_field_labels(text)
    if not reason:
        reason = "Not specified."

    inferred_regions = infer_regions(f"{regions}\n{reason}\n{text}")
    if inferred_regions:
        merged = canonicalize_regions(split_regions(regions))
        existing = {item.lower() for item in merged}
        for region in inferred_regions:
            if region.lower() not in existing:
                merged.append(region)
                existing.add(region.lower())
        regions = ", ".join(merged)
    else:
        regions = ", ".join(canonicalize_regions(split_regions(regions)))
    if not regions:
        regions = "Other"
    regions = drop_unneeded_background(regions, reason)
    regions = drop_unneeded_lymphocytes(regions, reason)

    magnification = normalize_magnification(magnification, f"{text}\n{reason}")

    return (
        f"Region(s): {regions}\n"
        f"Magnification: {magnification}\n"
        f"Reason: {reason}"
    )


def parse_response_fields(text):
    label_pattern = re.compile(
        r"(?i)(region\s*\(\s*s\s*\)|regions?|tissue\s*\(\s*s\s*\)|"
        r"magnification|reason)\s*(?:[-:：])?"
    )
    labels = []
    for match in label_pattern.finditer(text):
        raw_label = re.sub(r"\s+", "", match.group(1).lower())
        if raw_label in {"region(s)", "regions", "region", "tissue(s)"}:
            key = "regions"
        elif raw_label == "magnification":
            key = "magnification"
        else:
            key = "reason"
        labels.append((key, match.start(), match.end()))

    fields = {}
    for idx, (key, _, value_start) in enumerate(labels):
        value_end = labels[idx + 1][1] if idx + 1 < len(labels) else len(text)
        value = clean_field_value(text[value_start:value_end])
        if value and key not in fields:
            fields[key] = value
    return fields


def remove_field_labels(text):
    text = re.sub(
        r"(?i)(region\s*\(\s*s\s*\)|regions?|tissue\s*\(\s*s\s*\)|"
        r"magnification|reason)\s*(?:[-:：])?",
        "",
        text,
    )
    return clean_field_value(text)


def clean_field_value(value):
    value = value.replace("_", " ")
    value = re.sub(r"^[\s\-:]+", "", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;")


def split_regions(regions):
    regions = re.sub(r"(?i)^tissue\(s\)\s*[-:]\s*", "", regions.strip())
    regions = re.sub(r"(?i)\b(tissue\(s\)|region\(s\))\b", "", regions)
    return [item.strip() for item in re.split(r"[,;/，]", regions) if item.strip()]


def canonicalize_regions(regions):
    aliases = {
        "tumor": "Tumor",
        "tumour": "Tumor",
        "benign epithelium": "Benign epithelium",
        "stroma": "Stroma",
        "necrosis": "Necrosis",
        "lymphocytes": "Lymphocytes",
        "lymphocyte": "Lymphocytes",
        "blood vessel": "Blood vessel",
        "vessel": "Blood vessel",
        "nerve": "Nerve",
        "adipose tissue": "Adipose tissue",
        "smooth muscle": "Smooth muscle",
        "inflammation": "Inflammation",
        "hemorrhage": "Hemorrhage",
        "haemorrhage": "Hemorrhage",
        "background": "Background",
        "other": "Other",
    }
    canonical = []
    seen = set()
    for region in regions:
        key = clean_field_value(region).lower()
        key = re.sub(r"(?i)\b(tissues?|regions?)\b", "", key).strip(" -")
        value = aliases.get(key)
        if value and value.lower() not in seen:
            canonical.append(value)
            seen.add(value.lower())
    return canonical


def drop_unneeded_background(regions, reason):
    if re.search(r"(?i)background|artifact|artefact|non[- ]?tissue", reason):
        return regions
    filtered = [
        region.strip()
        for region in regions.split(",")
        if region.strip().lower() != "background"
    ]
    return ", ".join(filtered) or "Other"


def drop_unneeded_lymphocytes(regions, reason):
    if re.search(r"(?i)lymphocytes?|lymphocytic|inflammation|inflammatory", reason):
        return regions
    filtered = [
        region.strip()
        for region in regions.split(",")
        if region.strip().lower() != "lymphocytes"
    ]
    return ", ".join(filtered) or "Other"


def infer_regions(text):
    text = text.lower()
    inferred = []
    keyword_map = (
        ("Tumor", ("tumor", "malignant", "carcinoma", "neoplasm")),
        ("Nerve", ("nerve", "perineural", "perineurium", "perineurial")),
        ("Stroma", ("stroma", "stromal")),
        ("Necrosis", ("necrosis", "necrotic")),
        ("Lymphocytes", ("lymphocyte", "lymphocytic")),
        ("Inflammation", ("inflammation", "inflammatory")),
        ("Blood vessel", ("blood vessel", "vascular", "vessel")),
        ("Hemorrhage", ("hemorrhage", "haemorrhage")),
    )
    for region, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            inferred.append(region)
    if "background" in text and re.search(r"background|artifact|artefact|non[- ]?tissue", text):
        inferred.append("Background")
    return inferred


def normalize_magnification(magnification, text):
    matches = re.findall(r"\b(5|10|20|40|44)\s*[xX×]\b", f"{magnification}\n{text}")
    if matches:
        normalized = []
        for value in matches:
            if value == "44":
                value = "40"
            formatted = f"{value}x"
            if formatted not in normalized:
                normalized.append(formatted)
        return ", ".join(normalized)
    bare_matches = re.findall(r"\b(5|10|20|40|44)\b", magnification)
    if bare_matches:
        value = bare_matches[0]
        if value == "44":
            value = "40"
        return f"{value}x"
    if re.search(r"(?i)nerve|perineural|perineur", text):
        return "40x"
    return magnification or "20x"


def guard_unsupported_conclusions(response, question):
    question_lower = question.lower()

    if "perineural invasion" in question_lower or "pni" in question_lower:
        return (
            "Region(s): Tumor, Nerve\n"
            "Magnification: 10x, 40x\n"
            "Reason: Tumor and nerve regions should be examined to plan "
            "assessment of tumor-nerve interaction; 10x helps locate nerves in "
            "their tissue context and 40x helps inspect perineural or "
            "intraneural tumor involvement."
        )
    if "lymphovascular invasion" in question_lower or "lvi" in question_lower:
        return (
            "Region(s): Tumor, Blood vessel\n"
            "Magnification: 10x, 40x\n"
            "Reason: Tumor and vascular spaces should be examined to plan "
            "assessment of tumor emboli within vessel-like spaces; 10x provides "
            "architectural context and 40x helps confirm endothelial-lined "
            "spaces and intraluminal tumor cells."
        )
    if "cribriform" in question_lower:
        return (
            "Region(s): Tumor\n"
            "Magnification: 10x, 20x\n"
            "Reason: Tumor regions should be examined because cribriform growth "
            "is a malignant glandular architecture pattern; 10x provides gland "
            "pattern context and 20x helps assess punched-out lumina and fused "
            "glandular structures."
        )
    if "necrosis" in question_lower or "necrotic" in question_lower:
        return (
            "Region(s): Tumor, Necrosis\n"
            "Magnification: 10x, 20x\n"
            "Reason: Tumor and necrotic areas should be examined to plan "
            "assessment of necrosis within malignant tissue; 10x localizes "
            "necrotic foci and 20x helps distinguish true necrosis from debris "
            "or artifact."
        )
    if (
        "tumor-infiltrating lymphocytes" in question_lower
        or "tumour-infiltrating lymphocytes" in question_lower
        or "til" in question_lower
    ):
        return (
            "Region(s): Tumor, Lymphocytes, Stroma\n"
            "Magnification: 10x, 20x, 40x\n"
            "Reason: Tumor, lymphocytes, and adjacent stroma should be examined "
            "to plan assessment of immune infiltration at the tumor interface; "
            "10x estimates distribution, 20x assesses infiltration pattern, and "
            "40x confirms lymphocyte morphology."
        )
    if "gleason" in question_lower:
        return (
            "Region(s): Tumor\n"
            "Magnification: 20x, 40x\n"
            "Reason: Tumor regions should be examined because Gleason grading "
            "depends on malignant glandular architecture; 20x supports pattern "
            "assessment and 40x helps confirm cytologic and gland-forming "
            "details."
        )
    if "recurrence risk" in question_lower:
        return (
            "Region(s): Tumor, Stroma\n"
            "Magnification: 10x\n"
            "Reason: Tumor and adjacent stroma should be examined to plan "
            "assessment of tumor extent, invasion, and margin-related features; "
            "10x provides architectural context across larger tissue areas."
        )
    return response


def generate_reply(model, tokenizer, messages, args, question):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": get_eos_token_ids(tokenizer),
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )

    new_tokens = generated_ids[0][model_inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return clean_response(response, question)


def main():
    configure_utf8_output()
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print("\nModel loaded. Type your message and press Enter.")
    print("Commands: /exit or /quit to stop, /clear to reset history.")
    if args.keep_history:
        print("History mode: on\n")
    else:
        print("History mode: off; each question is planned independently.\n")

    messages = [{"role": "system", "content": args.system_prompt}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": args.system_prompt}]
            print("History cleared.\n")
            continue

        user_input = clean_user_input(user_input)
        if not user_input:
            print("Please enter only your question, not previous Qwen output.\n")
            continue

        if args.keep_history:
            messages.append({"role": "user", "content": user_input})
            response = generate_reply(model, tokenizer, messages, args, user_input)
            messages.append({"role": "assistant", "content": response})
        else:
            current_messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_input},
            ]
            response = generate_reply(model, tokenizer, current_messages, args, user_input)

        print(f"Qwen: {response}\n")


if __name__ == "__main__":
    main()
