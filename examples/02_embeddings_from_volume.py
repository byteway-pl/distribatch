# Databricks notebook source
# MAGIC %md
# MAGIC # Embed document chunks with a self-hosted model from a UC Volume
# MAGIC GPU cluster with Databricks Runtime ML. Uses vLLM when available, else transformers.

# COMMAND ----------

# MAGIC %pip install "distribatch[vllm,transformers]"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import distribatch as dib

embedder = dib.EmbedJob(
    model=dib.models.path("/Volumes/main/ml/weights/bge-small-en-v1.5", copy_to_local=True),
    text="{chunk_text}",
    normalize=True,
)

chunks = spark.table("main.rag.chunks")
vectors = embedder.run(chunks, backend="spark", batch_size=256, output_col="embedding")
vectors.write.mode("overwrite").saveAsTable("main.rag.chunk_embeddings")
print(dib.summarize(spark.table("main.rag.chunk_embeddings"), output_col="embedding"))
