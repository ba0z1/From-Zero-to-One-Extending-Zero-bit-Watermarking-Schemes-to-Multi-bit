# QuadWrite-1000

QuadWrite-1000 contains 1,000 English writing prompts for long-form generation. Stories, travelogues, dialogues, and explanatory essays each account for 250 prompts. Each prompt asks for approximately 1,800 words.

The collection starts with 25 original prompts per genre. Each genre adds 225 prompts by pairing 45 topics with five writing angles. The `topic_id` field groups variants of a topic and can be used to keep them together when splitting the data.

## Use the prompts

Download the prompts as [JSONL](data/prompts.jsonl), [JSON](data/prompts.json), or [CSV](data/prompts.csv). All three files contain the same records. Use the `prompt` field as the user message and `task_id` to identify the record.

```python
import json
from pathlib import Path

prompts = json.loads(Path("data/prompts.json").read_text(encoding="utf-8"))
print(prompts[0]["task_id"], prompts[0]["prompt"])
```

## Reproduce the dataset

The [construction files](construction) contain the original prompts, topic lists, angles, and shared instructions. From this directory, run the following commands with Python 3.

```sh
python3 scripts/build_dataset.py
python3 scripts/validate_dataset.py
```

## License

The prompts, construction data, and metadata are licensed under [CC BY 4.0](LICENSE). Credit the QuadWrite-1000 contributors, link to the [project](https://github.com/ba0z1/Let_Watermarks_Speak) and [license](https://creativecommons.org/licenses/by/4.0/), and indicate any changes. The Python scripts use the repository's [MIT License](../../LICENSE).
