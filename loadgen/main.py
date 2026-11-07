"""Entry point: `slotgw-loadgen --scenario config/scenarios/mixed.yaml --seed 1`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from .plan import build_plan
from .runner import LoadRunner
from .scenario import load_scenario


def main() -> None:
    p = argparse.ArgumentParser(description="slotgw load generator (docs/DESIGN.md section 9)")
    p.add_argument("--scenario", required=True, help="scenario YAML (config/scenarios/*.yaml)")
    p.add_argument("--seed", type=int, default=1, help="master seed; same seed = same request sequence")
    p.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://localhost:8080"))
    p.add_argument("--runs-dir", default=os.environ.get("SLOTGW_RUNS_DIR", "runs"))
    p.add_argument("--config-dir", default=os.environ.get("SLOTGW_CONFIG_DIR", "config"))
    p.add_argument("--load-run-id", default=None, help="default: <scenario>[-label]-s<seed>-<timestamp>")
    p.add_argument("--label", default=None, help="tag for the gateway settings under test (e.g. oc1.5); keeps runs apart in the report")
    p.add_argument("--no-reservations", action="store_true", help="ignore the scenario's reservation specs (on-demand only)")
    p.add_argument("--reservation-replica", default=None, metavar="ID|none",
                   help="pin every reservation to this replica, or 'none' to ignore the scenario's `replica:` pins")
    p.add_argument("--plan-only", action="store_true", help="print the plan as JSON and exit")
    p.add_argument("--log-level", default=os.environ.get("LOADGEN_LOG_LEVEL", "info"))
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise here
    scenario = load_scenario(args.scenario)
    plan = build_plan(scenario, args.seed, scenario_path=args.scenario)
    if args.plan_only:
        print(plan.to_json())
        return
    runner = LoadRunner(
        scenario,
        plan,
        gateway_url=args.gateway,
        runs_dir=args.runs_dir,
        load_run_id=args.load_run_id,
        config_dir=args.config_dir,
        scenario_path=args.scenario,
        label=args.label,
        reservations=not args.no_reservations,
        reservation_replica=args.reservation_replica,
    )
    meta = asyncio.run(runner.run())
    print(json.dumps({k: meta[k] for k in ("load_run", "gateway_run_id", "plan_digest", "counters", "elapsed_s")}, indent=1))
    sys.exit(0 if meta["counters"]["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
