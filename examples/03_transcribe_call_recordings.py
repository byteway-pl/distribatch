# Databricks notebook source
# MAGIC %md
# MAGIC # Transcribe call recordings with Whisper from Unity Catalog
# MAGIC GPU cluster. The model is staged once on a Volume, so executors need no registry
# MAGIC credentials. Results are checkpointed per file path.

# COMMAND ----------

# MAGIC %pip install "distribatch[whisper,transformers,audio,mlflow]"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import distribatch as dib

asr = dib.TranscribeJob(
    model=dib.models.uc(
        "main.audio.whisper_large_v3", alias="prod", stage_dir="/Volumes/main/audio/stage"
    ),
    audio_col="path",
    language="en",
    timestamps=True,
    engine_options={"compute_type": "float16", "vad_filter": True},
)

recordings = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.{wav,mp3,flac}")
    .load("/Volumes/main/raw/calls/")
    .select("path")
)
transcripts = asr.run(
    recordings,
    backend="spark",
    batch_size=8,
    output_col="transcript",
    checkpoint="/Volumes/main/audio/checkpoints/calls",
    id_col="path",
)
transcripts.write.mode("overwrite").saveAsTable("main.audio.call_transcripts")
print(dib.summarize(transcripts, output_col="transcript"))
