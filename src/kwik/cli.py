"""Command line: kwik <step> --config configs/<file>.yaml"""

from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="kwik", description="Does It Know What It Knows?")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [
        ("collect", "run the model, score answers, save hidden states"),
        ("baselines", "stated confidence + sampling agreement"),
        ("train", "probes + comparison with baselines"),
        ("report", "write results.md and plots"),
        ("all", "collect, baselines, train, report"),
        ("handcheck-export", "write 200 answers to label by hand"),
        ("handcheck-agree", "compare hand labels with automatic labels"),
        ("bench", "measure the probe's added latency"),
        ("serve", "start the FastAPI confidence API"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        if name == "serve":
            sp.add_argument("--port", type=int, default=8000)
        if name == "bench":
            sp.add_argument("--n", type=int, default=30)
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd in ("collect", "all"):
        from .collect import collect

        collect(cfg)
    if a.cmd in ("baselines", "all") and cfg.run_baselines:
        from .baselines import run_baselines

        run_baselines(cfg)
    if a.cmd in ("train", "all"):
        from .probes import train_and_evaluate

        print(json.dumps(train_and_evaluate(cfg.run_dir), indent=2))
    if a.cmd in ("report", "all"):
        from .report import write_report

        print(f"wrote {write_report(cfg.run_dir)}")
    if a.cmd == "handcheck-export":
        from .handcheck import export

        print(f"wrote {export(cfg.run_dir)}; fill in human_correct with 1 or 0")
    if a.cmd == "handcheck-agree":
        from .handcheck import agree

        print(json.dumps(agree(cfg.run_dir), indent=2))
    if a.cmd == "bench":
        import pandas as pd

        from .service import bench

        qs = pd.read_json(cfg.run_dir / "answers.jsonl", lines=True)
        qs = qs[qs.split == "test"]["question"].head(a.n).tolist()
        print(json.dumps(bench(cfg.run_dir, qs, cfg.device), indent=2))
    if a.cmd == "serve":
        import uvicorn

        from .service import make_app

        uvicorn.run(make_app(cfg.run_dir, cfg.device), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
