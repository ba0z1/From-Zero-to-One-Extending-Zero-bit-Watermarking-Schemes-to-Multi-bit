# From Zero to One: Extending Zero-bit Watermarking Schemes to Multi-bit

The generator embeds a fingerprint of the prompt, then fingerprints successive spans of generated tokens. A verifier recovers these fingerprints without loading the language model and reports how many response tokens it can authenticate.

## QuadWrite-1000

QuadWrite-1000 contains 1,000 English writing prompts, with 250 each for stories, travelogues, dialogues, and explanatory essays. [View the prompts (JSONL)](datasets/quadwrite-1000/data/prompts.jsonl) or [browse all formats](datasets/quadwrite-1000/).

## Run

Install the dependencies and open the Gradio demo.

```sh
python -m pip install -r requirements.txt
python main.py --m Qwen/Qwen2-0.5B-Instruct
```

To generate and verify a sample in the terminal, run both commands with the same key. Verification uses the token IDs saved in the JSON file.

```sh
export LWS_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python main.py generate --m Qwen/Qwen2-0.5B-Instruct --prompt "Write a travelogue about Kyoto." --output sample.json
python main.py verify --input sample.json
```

For a custom bit string, add `--payload 0101` to generation and recover it with `python main.py extract --input sample.json`.

The code uses the [MIT License](LICENSE). QuadWrite-1000 uses [CC BY 4.0](datasets/quadwrite-1000/LICENSE).
