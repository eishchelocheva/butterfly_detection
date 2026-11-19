#!/usr/bin/env python3
"""Create a review project and closed pools; default is an offline plan."""
import argparse
import copy
import json
from pathlib import Path
import sys

REVIEW = Path(__file__).resolve().parent
BBOX = REVIEW.parent / "bbox"
sys.path.insert(0, str(BBOX))
import create_project as bbox_project
from tasks_api import ApiClient, ApiError, ProvisioningError, apply_plan


def build_plan(settings, interface, lock, instructions, examples_base_url=None,
               with_training=False, now=None):
    # The shared instruction renderer uses bbox/instructions/examples.json.
    # It is the source for the identical examples copied into review.
    steps = bbox_project.build_plan(
        settings, interface, lock, instructions, examples_base_url, with_training, now
    )
    spec = steps[0]["payload"]["task_spec"]
    spec["input_spec"] = {
        "photo_id": {"type": "string", "required": True},
        "image_url": {"type": "url", "required": True},
        "boxes": {"type": "json", "required": True},
    }
    spec["output_spec"] = {
        "review_verdict": {
            "type": "string", "required": True,
            "allowed_values": ["approve", "reject", "uncertain"],
        },
        "status": {
            "type": "string", "required": True,
            "allowed_values": ["objects", "empty", "uncertain"],
        },
        "boxes": {"type": "json", "required": True},
        "comment": {"type": "string", "required": False},
    }
    for step in steps:
        if step["role"] == "main":
            step["payload"]["private_name"] = "Основное ревью"
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REVIEW / "configs/api.json")
    parser.add_argument("--token", help="API token; overrides api_token from config")
    parser.add_argument("--interface", type=Path, default=REVIEW / "configs/interface.json")
    parser.add_argument("--lock", type=Path, default=REVIEW / "configs/interface.lock.json")
    parser.add_argument("--instructions", type=Path, default=REVIEW / "instructions/annotator.html")
    parser.add_argument("--examples-base-url", help="Public HTTPS directory with instruction JPEGs")
    parser.add_argument("--with-training", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--auth-scheme", choices=("OAuth", "ApiKey"), default="OAuth")
    parser.add_argument("--state", type=Path, default=REVIEW / "results/api_state.json")
    parser.add_argument("--adopt", action="append", default=[], metavar="ROLE=ID")
    args = parser.parse_args(argv)
    try:
        if args.adopt and not args.apply:
            raise ValueError("--adopt requires --apply and an existing pending resource")
        def read_json(path):
            return json.loads(path.read_text(encoding="utf-8"))
        settings = read_json(args.config)
        steps = build_plan(
            settings, read_json(args.interface), read_json(args.lock),
            args.instructions.read_text(encoding="utf-8"),
            args.examples_base_url, args.with_training,
        )
        if not args.apply:
            print(json.dumps({"mode": "dry-run", "steps": steps},
                             ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        token = (args.token if args.token is not None else settings["api_token"]).strip()
        if not token:
            raise ValueError("Provide --token or set api_token in the review config before using --apply")
        adopted = {}
        for item in args.adopt:
            role, sep, resource_id = item.partition("=")
            if not sep or not role or not resource_id or role in adopted:
                raise ValueError("--adopt expects unique ROLE=ID pairs")
            adopted[role] = resource_id
        state = apply_plan(
            steps, args.state, ApiClient(token, auth_scheme=args.auth_scheme),
            adopt=adopted or None,
        )
        print("Created/verified resources (pools CLOSED; tasks not uploaded):")
        for role, resource in state["resources"].items():
            print(f"  {role}: {resource['id']}")
        print(f"State: {args.state}")
        return 0
    except (OSError, ValueError, ApiError, ProvisioningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
