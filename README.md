# distribatch

Batch model inference over DataFrames: **LLM generation, text embeddings and speech
transcription**, on Databricks or anywhere Python runs.

```python
import distribatch as dib
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

job = dib.GenerateJob(
    model=dib.models.databricks("databricks-meta-llama-3-3-70b-instruct"),
    prompt="Classify the sentiment of this review:\n{review}",
    output_schema=Sentiment,
)
scored = job.run(reviews_df, backend="spark", output_col="sentiment")
```

- **One job, three backends:** `local`, `multiprocessing` (one worker per GPU) and
  `spark` (`mapInArrow`, also on Spark Connect / serverless).
- **Any model source:** Unity Catalog registered models, Hugging Face weights in a
  directory or UC Volume, or any OpenAI-compatible API (OpenAI, Databricks Model Serving,
  vLLM servers, …).
- **Engines:** vLLM on GPUs, Hugging Face transformers on CPU/GPU, faster-whisper for
  speech, and an async OpenAI client with retries, rate limits and concurrency control.
- **Built for long runs:**
  - per-row errors instead of failed jobs;
  - structured output validated against pydantic models and typed as Spark structs;
  - checkpoints that resume where a run stopped;
  - deduplication of identical inputs;
  - run statistics with cost estimates.

## Installation

```bash
pip install "distribatch[openai]"                       # APIs and Databricks Model Serving
pip install "distribatch[vllm,mlflow]"                  # self-hosted LLMs/embedders from Unity Catalog
pip install "distribatch[whisper,transformers,audio]"   # local speech-to-text
pip install "distribatch[all]"
```

Python 3.10+. The core install is light: heavy dependencies come with the extras and are
imported only when used.

## Embeddings and transcription

```python
vectors = dib.EmbedJob(
    model=dib.models.path("/Volumes/main/ml/weights/bge-m3", copy_to_local=True),
    text="{title}\n\n{body}",
    normalize=True,
).run(docs_df, backend="spark", output_col="embedding")          # array<float>

calls = spark.read.format("binaryFile").load("/Volumes/main/raw/calls/").select("path")
transcripts = dib.TranscribeJob(
    model=dib.models.uc("main.audio.whisper_large_v3", alias="prod"),
    audio_col="path",
    timestamps=True,
).run(calls, backend="spark", output_col="transcript")
```

## Long runs

```python
out = job.run(
    reviews_df,
    backend="spark",
    checkpoint="/Volumes/main/ml/checkpoints/reviews",   # resume after interruptions
    id_col="review_id",
    dedupe=True,                                        # identical inputs sent once
)
stats = dib.summarize(out, output_col="sentiment")
print(stats)                                            # success rate, errors, tokens, latency
print(stats.cost(input_per_million=0.5, output_per_million=1.5))
```

On a single machine use `backend="multiprocessing"`. In plain Python scripts, put the
call under `if __name__ == "__main__":`, because worker processes are spawned.

## Documentation

The **[wiki](https://github.com/byteway-pl/distribatch/wiki)** covers every job, engine
option and backend, plus the output contract, checkpoints, a Databricks guide
(credentials, GPU clusters, Unity Catalog) and troubleshooting. Runnable examples are in
[`examples/`](examples/).

> **Status:** pre-1.0. The API may still change between minor versions, see
> [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0
