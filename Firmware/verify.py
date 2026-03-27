#!/usr/bin/env python3
"""
=============================================================
 IoT Fingerprinting -- Data Verification Script
 Works on Mac, Windows, Linux, and Raspberry Pi.
=============================================================
 USAGE:
   python3 verify.py                  # checks all CSVs in default folder
   python3 verify.py myfile.csv       # checks one specific file
   python3 verify.py /path/to/folder  # checks all CSVs in a folder

 Default folder:
   Mac/Windows : ~/Desktop/IoT_Data/
   RPi/Linux   : ~/data/
=============================================================
"""

import os
import sys
import glob
import csv
import platform
from datetime import datetime


def get_default_dir():
    system = platform.system()
    if system in ("Darwin", "Windows"):
        return os.path.join(os.path.expanduser("~"), "Desktop", "IoT_Data")
    else:
        return os.path.expanduser("~/data")


def verify_file(filepath):
    rows = []

    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check required columns exist before reading rows
            if reader.fieldnames is None or not all(
                c in reader.fieldnames for c in
                ["node_id", "board_timestamp_ms", "rssi", "lqi"]
            ):
                print("\n  File   : %s" % os.path.basename(filepath))
                print("  Status : ERROR -- missing expected CSV columns")
                print("  Found  : %s" % reader.fieldnames)
                return False

            for row in reader:
                try:
                    rows.append({
                        "ts_ms": int(row["board_timestamp_ms"]),
                        "rssi":  int(row["rssi"]),
                        "lqi":   int(row["lqi"]),
                        "node":  row["node_id"].strip()
                    })
                except (ValueError, KeyError):
                    continue

    except FileNotFoundError:
        print("\n  ERROR: File not found: %s" % filepath)
        return False
    except Exception as e:
        print("\n  ERROR reading %s: %s" % (filepath, e))
        return False

    if not rows:
        print("\n  File   : %s" % os.path.basename(filepath))
        print("  Status : WARNING -- no valid data rows found")
        return False

    # ---- Compute stats -----------------------------------------------
    total    = len(rows)
    t_start  = rows[0]["ts_ms"]
    t_end    = rows[-1]["ts_ms"]
    dur_s    = (t_end - t_start) / 1000.0
    rate     = total / dur_s if dur_s > 0 else 0

    rssis    = [r["rssi"] for r in rows]
    rssi_min = min(rssis)
    rssi_max = max(rssis)
    rssi_avg = sum(rssis) / len(rssis)

    nodes_seen = sorted(set(r["node"] for r in rows))

    # Count gaps > 250ms (a missed packet gap)
    gaps = sum(
        1 for i in range(1, len(rows))
        if rows[i]["ts_ms"] - rows[i - 1]["ts_ms"] > 250
    )
    loss_pct = (gaps / total * 100) if total > 0 else 0

    # Completeness only meaningful for 30-min main experiments, not range_test files
    expected_30min = 10 * 60 * 30   # 18000 packets expected
    is_main = "_main_" in os.path.basename(filepath)
    completeness = min(100.0, total / expected_30min * 100) if (is_main and dur_s > 60) else None

    ok     = rate >= 8.0 and loss_pct < 5.0 and rssi_avg > -90
    status = "OK" if ok else "WARNING"

    # ---- Print report ------------------------------------------------
    print("\n  " + "=" * 52)
    print("  File     : %s" % os.path.basename(filepath))
    print("  Status   : %s" % status)
    print("  " + "-" * 52)
    print("  Node IDs : %s" % ", ".join(nodes_seen))
    print("  Packets  : %d" % total)
    print("  Duration : %.1fs  (%.1f min)" % (dur_s, dur_s / 60))
    print("  Pkt rate : %.1f pkt/s  (target >= 10)" % rate)
    if completeness is not None:
        print("  Complete : %.1f%% of expected 30-min data" % completeness)
    print("  RSSI     : avg=%.1f  min=%d  max=%d  (dBm)"
          % (rssi_avg, rssi_min, rssi_max))
    print("  Gaps>250ms: %d  (%.1f%% loss)" % (gaps, loss_pct))

    if not ok:
        print("  " + "-" * 52)
        if rate < 8.0:
            print("  [!] Packet rate too low -- is the transmitter node powered on?")
        if loss_pct > 5.0:
            print("  [!] High packet loss -- nodes may be too far apart")
        if rssi_avg <= -90:
            print("  [!] Very low RSSI -- move nodes closer together")

    print("  " + "=" * 52)
    return ok


def main():
    print("\n" + "=" * 56)
    print("  IoT Fingerprinting -- Data Verification")
    print("  Platform : %s" % platform.system())
    print("  Time     : %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 56)

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            files = sorted(glob.glob(os.path.join(target, "*.csv")))
            if not files:
                print("\nNo CSV files found in: %s" % target)
                sys.exit(1)
        elif os.path.isfile(target):
            files = [target]
        else:
            print("\nERROR: '%s' is not a valid file or folder." % target)
            sys.exit(1)
    else:
        data_dir = get_default_dir()
        print("\nScanning: %s" % data_dir)
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not files:
            print("No CSV files found in: %s" % data_dir)
            print("\nUsage:")
            print("  python3 verify.py")
            print("  python3 verify.py nodeB_bridge_main_20260310.csv")
            print("  python3 verify.py ~/Desktop/IoT_Data/")
            sys.exit(1)

    print("\nFound %d file(s) to check." % len(files))

    ok_count      = 0
    warning_count = 0

    for filepath in files:
        result = verify_file(filepath)
        if result:
            ok_count += 1
        else:
            warning_count += 1

    print("\n" + "=" * 56)
    print("  SUMMARY: %d OK, %d WARNING" % (ok_count, warning_count))
    if warning_count > 0:
        print("  Re-check WARNING files before leaving the field!")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
