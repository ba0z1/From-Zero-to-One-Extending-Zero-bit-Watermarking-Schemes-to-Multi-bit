"""Generate and verify hash-chained language-model watermarks."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import secrets

import torch

from watermark import Config, Sender, derive_payload_key, derive_response_key, extract_payload, verify


FORMAT = "lws-six-layer-v1"
PAYLOAD_FORMAT = "lws-six-layer-payload-v1"


def master_key(ephemeral=False):
    value = os.environ.get("LWS_KEY")
    if value is None:
        if ephemeral:
            return secrets.token_bytes(32)
        raise ValueError("Set LWS_KEY to a 32-byte key encoded as 64 hexadecimal digits")
    key = bytes.fromhex(value)
    if len(key) != 32:
        raise ValueError("LWS_KEY must contain 32 bytes encoded as hexadecimal")
    return key


def encode_prompt(tokenizer, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a nonempty prompt")
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
    return tokenizer.encode(prompt, add_special_tokens=True)


def display_text(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class Generator:
    def __init__(self, model_name, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not model_name:
            raise ValueError("Provide a Hugging Face model ID or local path with --m")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.vocab_size = self.model.get_output_embeddings().weight.shape[0]
        self.eos = set()
        for source in [self.tokenizer, self.model.config, self.model.generation_config]:
            value = getattr(source, "eos_token_id", None)
            if value is not None:
                self.eos.update([value] if isinstance(value, int) else value)

    @torch.inference_mode()
    def generate(self, prompt, key, max_tokens, temperature, hash_bits, threshold_bits, payload=None):
        if max_tokens < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Token limit and temperature must be positive")
        ids = encode_prompt(self.tokenizer, prompt)
        config = Config(self.vocab_size, hash_bits, threshold_bits)
        nonce = secrets.token_bytes(16)
        response_key = derive_response_key(key, nonce) if payload is None else derive_payload_key(key, nonce)
        sender = Sender(ids, response_key, config, payload)
        inputs = torch.tensor([ids], device=self.device)
        cache = None
        released = []
        for _ in range(max_tokens):
            result = self.model(input_ids=inputs, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            probs = torch.softmax(result.logits[0, -1].float() / temperature, dim=-1)
            token = sender.embed_token(probs)
            if token in self.eos:
                break
            released.append(token)
            inputs = torch.tensor([[token]], device=self.device) if cache is not None else torch.tensor(
                [ids + released], device=self.device
            )
        record = dict(format=FORMAT, config=asdict(config), nonce=nonce.hex(),
                      prompt_ids=ids, response_ids=released, text=display_text(self.tokenizer, released))
        if payload is not None:
            record.update(format=PAYLOAD_FORMAT, payload_length=len(payload))
        return record


def check_record(record, expected_format, expected_config):
    if not isinstance(record, dict) or record.get("format") != expected_format:
        raise ValueError("Use verify for a hash-chain sample and extract for a custom payload")
    if Config(**record["config"]) != expected_config:
        raise ValueError("Sample settings differ from the verifier's vocabulary, hash length or threshold")


def verify_sample(record, key, window=8, *, expected_config, prompt_ids=None, response_ids=None):
    check_record(record, FORMAT, expected_config)
    response_key = derive_response_key(key, bytes.fromhex(record["nonce"]))
    return verify(record["prompt_ids"] if prompt_ids is None else prompt_ids,
                  record["response_ids"] if response_ids is None else response_ids,
                  response_key, expected_config, window=window)


def read_payload(record, key, window, expected_config, response_ids=None):
    check_record(record, PAYLOAD_FORMAT, expected_config)
    response_key = derive_payload_key(key, bytes.fromhex(record["nonce"]))
    return extract_payload(record["response_ids"] if response_ids is None else response_ids,
                           response_key, expected_config, record["payload_length"], window=window)


def verification_label(result):
    endpoint = result["authenticated_prefix_tokens"]
    if endpoint is None:
        scope = "The prompt and response have no authenticated prefix."
    elif endpoint == 0:
        scope = "The first link authenticates the prompt only."
    else:
        scope = f"The prompt and the first {endpoint} response tokens are authenticated."
    return f"{result['status']}. {scope}"


def serve(generator, key, host, port):
    import gradio as gr

    def generate_ui(prompt, length, temperature, hash_bits, threshold_bits, payload):
        record = generator.generate(prompt, key, int(length), temperature, int(hash_bits), int(threshold_bits),
                                    payload=payload or None)
        return record["text"], (prompt, record), "", "", ""

    def verify_ui(prompt, text, sample, mode):
        if not sample:
            raise gr.Error("Generate a response first.")
        original_prompt, record = sample
        tokenizer = generator.tokenizer
        ids = record["response_ids"] if text == record["text"] else tokenizer.encode(text, add_special_tokens=False)
        config = Config(generator.vocab_size, record["config"]["hash_bits"], record["config"]["threshold_bits"])
        window = 8 if mode == "Window" else 0
        if record["format"] == PAYLOAD_FORMAT:
            result = read_payload(record, key, window, config, ids)
            label = f"Recovered {result['recovered_count']} of {record['payload_length']} payload bits."
            return label, result["recovered_bits"], ""
        prompt_ids = record["prompt_ids"] if prompt == original_prompt else encode_prompt(tokenizer, prompt)
        result = verify_sample(record, key, window, expected_config=config,
                               prompt_ids=prompt_ids, response_ids=ids)
        endpoint = result["authenticated_prefix_tokens"]
        prefix = display_text(tokenizer, ids[:endpoint]) if endpoint else ""
        return verification_label(result), result["recovered_bits"], prefix

    with gr.Blocks() as demo:
        gr.Markdown("# Let Watermarks Speak")
        sample = gr.State()
        prompt = gr.Textbox(label="Prompt", value="What is watermarking?")
        payload = gr.Textbox(label="Custom payload bits (leave blank for hash-chain authentication)", value="")
        with gr.Row():
            length = gr.Slider(1, 8192, step=1, value=512, label="Maximum new tokens")
            temperature = gr.Slider(0.1, 1.5, step=0.05, value=1.0, label="Temperature")
        with gr.Row():
            hash_bits = gr.Slider(1, 128, step=1, value=16, label="Hash bits per link")
            threshold_bits = gr.Slider(10, 64, step=1, value=32, label="Stopping threshold (bits)")
        generate_button = gr.Button("Generate")
        response = gr.Textbox(label="Response to verify", lines=6, interactive=True)
        mode = gr.Radio(["Sequential", "Window"], value="Window", label="Receiver")
        verify_button = gr.Button("Verify or extract")
        status = gr.Textbox(label="Verification", interactive=False)
        bits = gr.Textbox(label="Recovered bits", interactive=False)
        prefix = gr.Textbox(label="Authenticated response prefix", interactive=False)
        generate_button.click(generate_ui, [prompt, length, temperature, hash_bits, threshold_bits, payload],
                              [response, sample, status, bits, prefix])
        verify_button.click(verify_ui, [prompt, response, sample, mode], [status, bits, prefix])
    demo.launch(server_name=host, server_port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "generate", "verify", "extract"), nargs="?", default="serve")
    parser.add_argument("-m", "--m", "--model", dest="model", help="Hugging Face model ID or local model path")
    parser.add_argument("-server_name", "--server_name", default="127.0.0.1")
    parser.add_argument("-server_port", "--server_port", type=int, default=7860)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--prompt")
    parser.add_argument("--payload", help="Embed these binary digits in place of hash-chain authentication")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hash-bits", type=int, default=16)
    parser.add_argument("--threshold-bits", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, help="Expected model output vocabulary size for verification")
    parser.add_argument("--output", type=Path, help="Write a generated JSON sample")
    parser.add_argument("--input", type=Path, help="Read a JSON sample for token-ID verification")
    parser.add_argument("--window", type=int, choices=(0, 8), default=8, help="0 for sequential, 8 for window recovery")
    args = parser.parse_args()
    try:
        key = master_key(ephemeral=args.action == "serve")
        if args.action in ("verify", "extract"):
            if args.input is None:
                parser.error(f"{args.action} requires --input")
            record = json.loads(args.input.read_text(encoding="utf-8"))
            vocab = args.vocab_size if args.vocab_size is not None else record["config"]["vocab_size"]
            if args.vocab_size is None and (type(vocab) is not int or vocab > 262144):
                raise ValueError("For a vocabulary larger than 262144, supply its expected size with --vocab-size")
            config = Config(vocab, args.hash_bits, args.threshold_bits)
            if args.action == "verify":
                result = verify_sample(record, key, args.window, expected_config=config)
            else:
                result = read_payload(record, key, args.window, config)
            print(json.dumps(result, indent=2))
            return
        if args.action == "generate" and (not args.prompt or args.output is None):
            parser.error("generate requires --prompt and --output")
        generator = Generator(args.model, args.device)
        if args.action == "serve":
            serve(generator, key, args.server_name, args.server_port)
        else:
            record = generator.generate(args.prompt, key, args.max_new_tokens, args.temperature,
                                        args.hash_bits, args.threshold_bits, args.payload)
            args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            print(record["text"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
