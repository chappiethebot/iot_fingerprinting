#!/usr/bin/env python3
"""
=============================================================
 IoT Fingerprinting -- Range Test Logger
 Works on Mac, Linux, Windows, and Raspberry Pi.
=============================================================
 USAGE:
   python3 range_test_logger.py

 Tests 10 distances automatically (2 min each).
 Saves a results table at the end.

 Output folder:
   Mac/Windows : ~/Desktop/IoT_Data/
   RPi/Linux   : ~/data/
=============================================================
"""

import serial
import serial.tools.list_ports
import os
import sys
import time
import platform
from datetime import datetime

BAUD_RATE      = 115200
TEST_DISTANCES = [10, 20, 30, 40, 50, 60, 70, 80, 100, 120]  # metres
TEST_DURATION  = 120   # seconds per distance


def get_log_dir():
    system = platform.system()
    if system in ("Darwin", "Windows"):
        return os.path.join(os.path.expanduser("~"), "Desktop", "IoT_Data")
    else:
        return os.path.expanduser("~/data")


def find_port():
    all_ports = list(serial.tools.list_ports.comports())
    system    = platform.system()

    if not all_ports:
        print("ERROR: No serial ports found. Check USB cable.")
        sys.exit(1)

    print("\nAvailable ports:")
    for i, p in enumerate(all_ports):
        print("  [%d]  %-35s %s" % (i, p.device, p.description))

    print()
    if system == "Darwin":
        print("HINT (Mac): Pick the port with 'usbmodem' in the name.")
        print("            If multiple usbmodem ports, pick the FIRST (lower number).")
    elif system == "Windows":
        print("HINT (Windows): Pick the COM port for your board.")
        print("                Check Device Manager -> Ports (COM & LPT) if unsure.")
    else:
        print("HINT (Linux/RPi): Pick the /dev/ttyACM* port.")

    print()
    while True:
        choice = input("Enter port number [0-%d] or type port name manually: "
                       % (len(all_ports) - 1)).strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(all_ports):
                selected = all_ports[idx].device
                print("Selected: %s" % selected)
                return selected
            else:
                print("Enter a number between 0 and %d" % (len(all_ports) - 1))
        else:
            return choice


def collect(ser, duration_sec, first_run):
    """
    Collect RSSI packets for duration_sec seconds.
    first_run=True : skip one line (board CSV header printed on startup).
    first_run=False: do NOT skip -- would discard real data on 2nd+ distance.
    """
    rssi_vals  = []
    timestamps = []
    errors     = 0

    ser.reset_input_buffer()

    if first_run:
        ser.readline()   # discard board's CSV header line

    deadline = time.time() + duration_sec

    while time.time() < deadline:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("node_id"):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                errors += 1
                continue
            rssi = int(parts[2])
            ts   = int(parts[1])
            rssi_vals.append(rssi)
            timestamps.append(ts)
            remaining = deadline - time.time()
            print("  RSSI: %4d dBm | Packets: %4d | Remaining: %.0fs   "
                  % (rssi, len(rssi_vals), remaining),
                  end="\r", flush=True)
        except (ValueError, UnicodeDecodeError):
            errors += 1

    gaps = sum(
        1 for i in range(1, len(timestamps))
        if timestamps[i] - timestamps[i - 1] > 250
    )
    return rssi_vals, gaps


def main():
    LOG_DIR = get_log_dir()

    print("\n" + "=" * 60)
    print("  IoT Fingerprinting -- Range Test Logger")
    print("  Platform: %s" % platform.system())
    print("=" * 60)

    port = find_port()

    # Connection test
    print("\nTesting connection to %s..." % port)
    try:
        test_ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(1)
        test_ser.close()
        print("Connection OK.")
    except serial.SerialException as e:
        print("ERROR: Cannot open %s: %s" % (port, e))
        sys.exit(1)

    print("\nEnvironment options: bridge | garden | forest | river | lake")
    environment = input("Enter environment name: ").strip().lower()
    if environment not in ["bridge", "garden", "forest", "river", "lake"]:
        print("Warning: '%s' not a standard name. Continuing." % environment)

    while True:
        rx_label = input("Which node is the RECEIVER (connected to this device)? (A / B / C): ").strip().upper()
        if rx_label in ["A", "B", "C"]:
            break
        print("  Invalid input. Please enter A, B, or C.")

    while True:
        tx_label = input("Which node is the TRANSMITTER (powered by power bank)? (A / B / C): ").strip().upper()
        if tx_label in ["A", "B", "C"]:
            if tx_label == rx_label:
                print("  Transmitter and receiver cannot be the same node. Try again.")
            else:
                break
        else:
            print("  Invalid input. Please enter A, B, or C.")

    os.makedirs(LOG_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_file = os.path.join(
        LOG_DIR,
        "node%s_range_test_%s_%s.txt" % (rx_label, environment, ts)
    )

    results   = []
    first_run = True   # only skip header on very first distance

    print("\nTesting %d distances: %s metres" % (len(TEST_DISTANCES), TEST_DISTANCES))
    print("2 min per distance -- total approx %d minutes\n" % (len(TEST_DISTANCES) * 2))

    ser = None
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)   # wait for board to stabilise

        for dist in TEST_DISTANCES:
            print("\n" + "-" * 60)
            print("  DISTANCE: %d metres" % dist)
            print("-" * 60)
            input("  Place Node %s at %dm then press ENTER to start..." % (tx_label, dist))

            print("  Collecting for %ds...\n" % TEST_DURATION)
            rssi_vals, gaps = collect(ser, TEST_DURATION, first_run)
            first_run = False
            print()

            if rssi_vals:
                total    = len(rssi_vals)
                loss_pct = round(gaps / total * 100, 1)
                avg      = round(sum(rssi_vals) / total, 1)
                rate     = round(total / TEST_DURATION, 1)
                usable   = "USABLE" if loss_pct < 10 else "TOO FAR"
            else:
                total = 0; loss_pct = 0; avg = 0; rate = 0
                usable = "NO SIGNAL"

            results.append({
                "dist": dist, "total": total, "rate": rate,
                "avg": avg, "loss": loss_pct, "usable": usable
            })

            if rssi_vals:
                print("  Packets : %d  (%.1f pkt/s)" % (total, rate))
                print("  RSSI avg: %.1f dBm" % avg)
                print("  Loss    : %.1f%%" % loss_pct)
            else:
                print("  Packets : 0")
                print("  RSSI avg: N/A")
                print("  Loss    : N/A")
            print("  Status  : %s" % usable)

            if usable == "NO SIGNAL":
                cont = input("\n  No signal. Stop testing? (y/n): ").strip().lower()
                if cont == "y":
                    break

    except KeyboardInterrupt:
        print("\n\nStopped early by user.")
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    if not results:
        print("No results collected. Exiting.")
        sys.exit(0)

    # ---- Build and save results table --------------------------------
    usable_dists = [r["dist"] for r in results if r["usable"] == "USABLE"]
    max_dist     = usable_dists[-1] if usable_dists else "N/A"

    lines = [
        "",
        "=" * 72,
        "  RANGE TEST -- Receiver: Node %s | Transmitter: Node %s -- %s   |   %s"
        % (rx_label, tx_label, environment.upper(),
           datetime.now().strftime("%Y-%m-%d %H:%M")),
        "=" * 72,
        "  %-10s %-10s %-10s %-14s %-10s Status"
        % ("Dist(m)", "Packets", "Rate/s", "RSSI avg", "Loss%"),
        "  " + "-" * 68,
    ]

    for r in results:
        icon = "OK" if r["usable"] == "USABLE" else "X"
        lines.append(
            "  %-10s %-10s %-10s %-14s %-10s [%s] %s"
            % (r["dist"], r["total"], r["rate"],
               str(r["avg"]) + "dBm",
               str(r["loss"]) + "%",
               icon, r["usable"])
        )

    lines += [
        "  " + "-" * 68,
        "  MAXIMUM DEPLOYMENT DISTANCE: %s metres" % max_dist,
        "=" * 72,
        ""
    ]

    output = "\n".join(lines)
    print(output)

    with open(table_file, "w", encoding="utf-8") as f:
        f.write(output)
    print("Table saved: %s\n" % table_file)


if __name__ == "__main__":
    main()
