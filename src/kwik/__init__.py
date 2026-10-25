"""kwik: Does It Know What It Knows?

Pipeline:
  collect   -> run an open model on QA questions, save answers, correctness and hidden states
  baselines -> stated confidence and sampling agreement (the slow / unreliable alternatives)
  train     -> logistic-regression probes per layer, layer picked on a validation split
  report    -> results table, transfer to a held-out dataset, plots
  serve     -> FastAPI endpoint returning answer + risk score + added latency
"""

__version__ = "0.1.0"
