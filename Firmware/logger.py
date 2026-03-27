#!/usr/bin/env python3
"""
=============================================================
 IoT Fingerprinting -- Raspberry Pi Data Logger
 Logs from two receiver nodes simultaneously via USB serial.
=============================================================
 USAGE:
   python3 logger.py

 You will be prompted to select which node is the transmitter.
 The two receiver labels are derived automatically:
   Node A transmits -> Node B and Node C receive
   Node B transmits -> Node A and Node C receive
   Node C transmits -> Node A and Node B receive

 OUTPUT (saved to ~/data/):
   node<X>_<env>_<type>_<timestamp>.csv  (one file per receiver)

 CSV FORMAT:
   node_id, pi_timestamp, board_timestamp_ms, rssi, lqi
=============================================================
"""

import serial
import serial.tools.list_ports
import threading
import csv
import os
import sys
import time
from datetime import datetime

BAUD_RATE   = 115200
LOG_DIR     = os.path.expanduser("~/data")
FLUSH_EVERY = 50

# Which two nodes receive, given who is transmitting
RECEIVER_MAP = {
    "A": ("B", "C"),
    "B": ("A", "C"),
    "C": ("A", "B"),
}


def pick_transmitter():
    """Ask the user which node is the transmitter and return the two receiver labels."""
    while True:
        tx = input("\nWhich node is the TRANSMITTER today? (A / B / C): ").strip().upper()
        if tx in RECEIVER_MAP:
            rx1, rx2 = RECEIVER_MAP[tx]
            print("  Transmitter : Node %s  (powered by power bank, NOT connected to RPi)" % tx)
            print("  Receivers   : Node %s and Node %s  (connected to RPi via USB)" % (rx1, rx2))
            return tx, rx1, rx2
        print("  Invalid input. Please enter A, B, or C.")


def list_and_pick_ports(label1, label2):
    """Show all detected serial ports and ask the user to assign them to the two receiver nodes."""
    all_ports = list(serial.tools.list_ports.comports())

    if not all_ports:
        print("ERROR: No serial ports found. Check USB cables.")
        sys.exit(1)

    print("\nAvailable ports:")
    for i, p in enumerate(all_ports):
        print("  [%d]  %-25s %s" % (i, p.device, p.description))

    print()
    print("HINT: Plug Node %s in FIRST, then Node %s." % (label1, label2))
    print("      On RPi: first board -> /dev/ttyACM0, second -> /dev/ttyACM1")
    print()

    def pick(label):
        while True:
            choice = input("Select port for Node %s [0-%d]: "
                           % (label, len(all_ports) - 1)).strip()
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(all_ports):
                    selected = all_ports[idx].device
                    print("  Node %s -> %s" % (label, selected))
                    return selected
                else:
                    print("  Enter a number between 0 and %d." % (len(all_ports) - 1))
            else:
                return choice   # manual port name entry

    port1 = pick(label1)
    port2 = pick(label2)

    if port1 == port2:
        print("\nERROR: Node %s and Node %s cannot use the same port!" % (label1, label2))
        sys.exit(1)

    return port1, port2


def log_node(port, node_label, environment, exp_type, stop_event, stats):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(
        LOG_DIR,
        "node%s_%s_%s_%s.csv" % (node_label, environment, exp_type, timestamp)
    )

    print("  [Node %s]  Port : %s" % (node_label, port))
    print("  [Node %s]  File : %s" % (node_label, filename))

    packet_count = 0
    error_count  = 0
    ser          = None

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "pi_timestamp",
                             "board_timestamp_ms", "rssi", "lqi"])
            f.flush()

            # Read first line -- could be board CSV header or first data line
            first_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if first_line and "node_id" not in first_line:
                # Looks like a real data line -- try to save it
                parts = first_line.split(",")
                if len(parts) == 4:
                    try:
                        int(parts[2])
                        int(parts[3])
                        writer.writerow([parts[0], "%.6f" % time.time(),
                                         parts[1], parts[2], parts[3]])
                        packet_count += 1
                        stats[node_label] = packet_count
                    except ValueError:
                        pass

            while not stop_event.is_set():
                try:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line or line.startswith("node_id"):
                        continue
                    parts = line.split(",")
                    if len(parts) != 4:
                        error_count += 1
                        continue
                    int(parts[2])
                    int(parts[3])

                    pi_ts = time.time()
                    writer.writerow([parts[0], "%.6f" % pi_ts,
                                     parts[1], parts[2], parts[3]])
                    packet_count += 1
                    stats[node_label] = packet_count

                    if packet_count % FLUSH_EVERY == 0:
                        f.flush()
                        os.fsync(f.fileno())   # force write to SD card

                except ValueError:
                    error_count += 1
                except Exception as e:
                    error_count += 1
                    if not stop_event.is_set():
                        print("\n  [Node %s] Parse error: %s" % (node_label, e))

    except serial.SerialException as e:
        print("\n  [Node %s] SERIAL ERROR: %s" % (node_label, e))
        print("  [Node %s] Check USB cable and port assignment." % node_label)
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    print("\n  [Node %s] Done -- %d packets saved, %d errors"
          % (node_label, packet_count, error_count))
    print("  [Node %s] File: %s" % (node_label, filename))


def status_printer(stop_event, stats, label1, label2, duration_sec):
    start = time.time()
    while not stop_event.is_set():
        elapsed   = time.time() - start
        remaining = max(0, duration_sec - elapsed)
        print("  [STATUS] Elapsed: %.1f min | Remaining: %.1f min | "
              "Node %s: %d pkts | Node %s: %d pkts     "
              % (elapsed / 60, remaining / 60,
                 label1, stats.get(label1, 0),
                 label2, stats.get(label2, 0)),
              end="\r", flush=True)
        time.sleep(10)


def main():
    print("\n" + "=" * 60)
    print("  IoT Fingerprinting Data Logger (Raspberry Pi)")
    print("=" * 60)

    tx_label, rx_label1, rx_label2 = pick_transmitter()
    port1, port2 = list_and_pick_ports(rx_label1, rx_label2)

    # ---- User input --------------------------------------------------
    print("\nEnvironment options: bridge | garden | forest | river | lake")
    environment = input("Enter environment name: ").strip().lower()
    if environment not in ["bridge", "garden", "forest", "river", "lake"]:
        print("Warning: '%s' not a standard name. Continuing." % environment)

    print("\nExperiment type options: range_test | main")
    exp_type = input("Enter experiment type: ").strip().lower()

    if exp_type == "main":
        duration_min = 30
        print("Duration: 30 minutes")
    else:
        try:
            duration_min = int(input("Enter duration in minutes: ").strip())
        except ValueError:
            print("Invalid input. Defaulting to 5 minutes.")
            duration_min = 5

    duration_sec = duration_min * 60
    os.makedirs(LOG_DIR, exist_ok=True)

    print("\n" + "-" * 60)
    print("  Transmitter  : Node %s" % tx_label)
    print("  Receivers    : Node %s and Node %s" % (rx_label1, rx_label2))
    print("  Environment  : %s" % environment)
    print("  Experiment   : %s" % exp_type)
    print("  Duration     : %d minutes" % duration_min)
    print("  Node %s port  : %s" % (rx_label1, port1))
    print("  Node %s port  : %s" % (rx_label2, port2))
    print("  Output dir   : %s" % LOG_DIR)
    print("-" * 60)
    input("\nPress ENTER to start (Ctrl+C to abort early)...\n")

    # ---- Start threads -----------------------------------------------
    stop_event = threading.Event()
    stats      = {rx_label1: 0, rx_label2: 0}

    thread1 = threading.Thread(
        target=log_node,
        args=(port1, rx_label1, environment, exp_type, stop_event, stats),
        daemon=True
    )
    thread2 = threading.Thread(
        target=log_node,
        args=(port2, rx_label2, environment, exp_type, stop_event, stats),
        daemon=True
    )
    status_thread = threading.Thread(
        target=status_printer,
        args=(stop_event, stats, rx_label1, rx_label2, duration_sec),
        daemon=True
    )

    thread1.start()
    thread2.start()
    time.sleep(1)
    status_thread.start()

    print("Logging started. Running for %d minutes...\n" % duration_min)

    try:
        time.sleep(duration_sec)
    except KeyboardInterrupt:
        print("\n\nStopped early by user.")

    stop_event.set()
    thread1.join(timeout=5)
    thread2.join(timeout=5)

    print("\n" + "=" * 60)
    print("  LOGGING COMPLETE")
    print("  Node %s : %d packets" % (rx_label1, stats[rx_label1]))
    print("  Node %s : %d packets" % (rx_label2, stats[rx_label2]))
    print("  Files  : %s" % LOG_DIR)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
