# Databricks notebook source
# MAGIC %md
# MAGIC # Classify support tickets with a Foundation Model endpoint
# MAGIC Structured output with a pydantic model, Spark backend, checkpoint and statistics.
# MAGIC Executors need `DATABRICKS_HOST` / `DATABRICKS_TOKEN` in the cluster environment.

# COMMAND ----------

# MAGIC %pip install "distribatch[openai]"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Literal

from pydantic import BaseModel

import distribatch as dib


class Route(BaseModel):
    team: Literal["billing", "technical", "account", "other"]
    urgent: bool
    summary: str


job = dib.GenerateJob(
    model=dib.models.databricks("databricks-meta-llama-3-3-70b-instruct"),
    system="You route customer support tickets. Be concise.",
    prompt="Subject: {subject}\n\n{body}",
    output_schema=Route,
    sampling=dib.Sampling(temperature=0, max_tokens=200),
    engine_options={"max_concurrency": 8},
)

tickets = spark.table("main.support.tickets")
print(job.preview(tickets, n=3)[["subject", "output", "output_error"]])

# COMMAND ----------

routed = job.run(
    tickets,
    backend="spark",
    partitions=8,
    output_col="route",
    checkpoint="/Volumes/main/support/checkpoints/ticket_routing",
    id_col="ticket_id",
)
routed.write.mode("overwrite").saveAsTable("main.support.tickets_routed")

stats = dib.summarize(routed, output_col="route")
print(stats)
print(f"estimated cost: ${stats.cost(input_per_million=0.5, output_per_million=1.5):.2f}")
