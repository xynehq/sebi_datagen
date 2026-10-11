# Model management

Discovery looks in:

- `models/adapters/`
- `models/cache/`
- `models/registry.json` (gitignored; removing an entry does not delete weights)
- directories passed with `--model-search-root`
- the Hugging Face cache (`HF_HOME`, otherwise `~/.cache/huggingface/hub`)

A directory is a model only when the layout is recognizable: a PEFT adapter (`adapter_config.json` plus adapter weights), a Transformers directory (`config.json` plus weights), an IndicOCR snapshot, or a PARSeq `.ckpt` file. A lone weight file is ignored.

`--model-path` is used as given. `--model-hf` downloads that repository into `models/cache` when the files are not already there. `--model-url` accepts a direct file URL. `--offline` does not download. Setup does not download weights.

A PEFT adapter records `base_model_name_or_path` and is not loaded as a standalone base model or as a PARSeq checkpoint. Language support is `unknown` unless the model metadata contains a language field.
