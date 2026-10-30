"""Entry point: `slotgw` (or `python -m slotgw.main`)."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import uvicorn

from .app import create_app
from .config import load_config
from .meter import Meter


def main() -> None:
    p = argparse.ArgumentParser(description="slotgw gateway")
    p.add_argument("--host", default=os.environ.get("SLOTGW_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SLOTGW_PORT", "8080")))
    p.add_argument("--config-dir", default=os.environ.get("SLOTGW_CONFIG_DIR", "config"))
    p.add_argument("--runs-dir", default=os.environ.get("SLOTGW_RUNS_DIR", "runs"))
    p.add_argument("--run-id", default=os.environ.get("SLOTGW_RUN_ID") or time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--log-level", default=os.environ.get("SLOTGW_LOG_LEVEL", "info"))
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config_dir)
    meter_path = Path(args.runs_dir) / args.run_id / "meter.sqlite"
    meter = Meter(meter_path, run_id=args.run_id)
    logging.getLogger("slotgw").info("run %s, meter at %s", args.run_id, meter_path)
    app = create_app(config, meter)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
