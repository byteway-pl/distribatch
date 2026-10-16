# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0a1] - 2026-10-16

First public pre-release, for end-to-end verification on Databricks. Install it
explicitly: `pip install "distribatch[...]==0.1.0a1"`.

### Added
- **Jobs:**
  - `GenerateJob`: `{column}` prompt templates, system prompt, sampling parameters,
    and structured output validated against pydantic models;
  - `EmbedJob`: text templates, optional `dimensions` and L2 normalization;
  - `TranscribeJob`: audio paths or bytes, language, initial prompt, segment
    timestamps, translation.
- **Model sources:**
  - Unity Catalog registered models via MLflow, with alias resolution and optional
    one-time staging on the driver (`stage_dir`);
  - Hugging Face weights in directories, UC Volumes or DBFS, with optional copy to
    local disk;
  - OpenAI-compatible APIs and Databricks Model Serving (`models.databricks`);
  - a `fake` model for dry runs.
- **Engines:**
  - `openai`: chat, embeddings and audio, with retries/backoff, concurrency and rate
    limits, native JSON-schema output with automatic fallback;
  - `vllm`: generation with structured outputs, and embeddings;
  - `transformers`: generation, embeddings honouring sentence-transformers pooling,
    and Whisper speech recognition;
  - `faster_whisper`: CTranslate2 Whisper, with automatic conversion of Hugging Face
    Whisper models.
- **Backends:**
  - `local`;
  - `multiprocessing`: one worker per GPU, recovery from crashed workers that isolates
    the offending row;
  - `spark`: `mapInArrow`, Spark Connect compatible, typed output schemas, GPU pinning
    per task.
- Per-row errors and metrics (`<out>_error`, `<out>_meta`) for every task and backend.
- Checkpoint/resume keyed by an id column and bound to the job's fingerprint;
  deduplication of identical inputs.
- `summarize()` / `RunStats`: success rate, errors by class, tokens, audio duration,
  latency percentiles, cost estimates and MLflow logging.
