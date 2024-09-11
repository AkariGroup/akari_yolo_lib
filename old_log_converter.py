#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timedelta

import ndjson

LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log_path",
        help="Log path to play",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    log_path = args.log_path
    new_log_path = log_path.replace(".json", "_converted.jsonl")
    print(new_log_path)
    log = None
    try:
        json_open = open(log_path, "r")
        log = json.load(json_open)
    except FileNotFoundError:
        print(f"Error: The file {log_path} does not exist.")
        return
    start_time = log["start_time"]
    interval = log["interval"]
    with open(new_log_path, mode="w", encoding="utf-8") as f:
        f.write("")
    for data in log["logs"]:
        new_data = {}
        new_data["time"] = (
            datetime.strptime(start_time, LOG_DATETIME_FORMAT)
            + timedelta(seconds=data["time"])
        ).strftime(LOG_DATETIME_FORMAT)
        new_data["interval"] = interval
        new_data["pos"] = data["pos"]
        new_data["id"] = data["id"]
        new_data["name"] = data["name"]
        with open(new_log_path, mode="a", encoding="utf-8") as f:
            writer = ndjson.writer(f)
            writer.writerow(new_data)


if __name__ == "__main__":
    main()
