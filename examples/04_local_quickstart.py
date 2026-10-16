"""Run anywhere: a dry run with the fake model, then an OpenAI model with local
multiprocessing. Needs `pip install "distribatch[openai]"` and OPENAI_API_KEY for step 2.
"""

import os

import pandas as pd
from pydantic import BaseModel

import distribatch as dib


class Sentiment(BaseModel):
    label: str
    confidence: float


reviews = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "review": ["Great!", "Broke after a day.", "It's fine.", "Great!"],
    }
)


def main() -> None:
    # 1. Dry run: checks the template and the output columns without any model.
    dry = dib.GenerateJob(model="fake", prompt="Sentiment of: {review}")
    print(dry.run(reviews)[["review", "output", "output_error"]])

    # 2. A real model, deduplicating identical reviews and checkpointing results.
    if os.environ.get("OPENAI_API_KEY"):
        job = dib.GenerateJob(
            model=dib.models.openai("gpt-4o-mini"),
            prompt="Classify the sentiment of this review:\n{review}",
            output_schema=Sentiment,
        )
        out = job.run(
            reviews, backend="multiprocessing", dedupe=True, checkpoint="ckpt", id_col="id"
        )
        print(out[["review", "output"]])
        print(dib.summarize(out))


if __name__ == "__main__":  # required for backend="multiprocessing" in scripts
    main()
