# IoT Radio Fingerprinting via Link Quality Fluctuations

Investigating whether manufacturing imperfections in IEEE 802.15.4 radio hardware produce stable, unique fingerprints that can identify individual sensor nodes and deployment environments using machine learning.

---

## Scientific Hypothesis

Due to manufacturing tolerances, each nRF52840 radio chip has microscopic imperfections that manifest as characteristic RSSI fluctuation patterns. This project tests two claims:

- **Scenario I (Environment ID):** RSSI fluctuations are shaped by the physical environment -> a classifier trained in one environment should **fail** on unseen environments  *confirms environment fingerprints do not generalise*
- **Scenario II (Node ID):** RSSI fluctuations carry a hardware fingerprint -> a classifier trained on data from known environments should **succeed** on an unseen environment  *confirms node fingerprints persist across environments*

---

## Hardware

| Item | Qty | Notes |
|---|---|---|
| Adafruit Feather Bluefruit Sense (nRF52840) | 3 | Nodes A, B, C |
| Raspberry Pi 3 Model B+ | 2 | Data loggers |
| USB-A to Micro-USB cables | 4 | RPi<->receiver board connections |
| Power bank (>=5 V, >=1 A) | 1 | Powers the transmitter node in the field |
| Smartphone | 1 | Wi-Fi hotspot for SSH access |
| Laptop | 1 | Arduino IDE flashing + SSH terminal |

**Radio protocol:** IEEE 802.15.4 @ 2425 MHz (Channel 15), +8 dBm TX power.

---

## Repository Structure

```
firmware/
|--transmitter_NodeA_NodeB_NodeC.ino          <- Flash to whichever node is transmitting
|--receiver_NodeA_B_and_C.ino       <- Flash to whichever nodes are receiving
|-- data_collection/
    |--logger.py                      <- RPi dual-node logger (main experiment)
    |--range_test_logger.py           <- Range test (run once before each site)
    |--verify.py                      <- Post-collection data quality check
|-- README.md

```
---

## Deployment Roles: Which Node Does What

All three nodes take turns as the **transmitter**. The other two are **receivers** connected to the RPis. The scripts ask you which node is transmitting at startup - no code edits needed.

| Session | Transmitter (power bank) | Receiver on RPi 1 | Receiver on RPi 2 |
|---|---|---|---|
| 1 | Node A | Node B | Node C |
| 2 | Node B | Node A | Node C |
| 3 | Node C | Node A | Node B |

Each session runs for **30 minutes x 5 environments = 2.5 hours per transmitter role.**
Total dataset: **3 roles x 5 environments x 2 receivers = 30 CSV files.**

---

## Firmware Setup

### Step 1 - Arduino IDE & Board Support

1. Download [Arduino IDE 2](https://www.arduino.cc/en/software).
2. Open **File - Preferences** and add to "Additional boards manager URLs":
   ```
   https://adafruit.github.io/arduino-board-index/package_adafruit_index.json
   ```
3. **Tools - Board - Boards Manager** - search **"Adafruit nRF52"** - Install.
4. **Sketch - Include Library - Manage Libraries** - search and install **Adafruit TinyUSB**.

### Step 2 - Flash the Transmitter (`transmitter_NodeA.ino`)

Open `firmware/transmitter_NodeA.ino`. Before flashing, update the node ID byte to match which physical node is acting as transmitter this session:

```cpp
// Inside loop() - change this line to match the transmitting node:
payload[0] = 0x01;   // 0x01 = Node A,  0x02 = Node B,  0x03 = Node C
```

Select **Board: Adafruit Feather nRF52840 Sense**, pick the correct COM/serial port, then **Upload**.

### Step 3 - Flash the Receivers (`receiver_NodeB_and_C.ino`)

Open `firmware/receiver_NodeB_and_C.ino`. Before flashing, set `NODE_ID` to match the physical board:

```cpp
#define NODE_ID  2    // 2 = Node B,  3 = Node C
// Change to 1 for Node A when it acts as a receiver
```

Flash each receiver board **separately** with its own correct `NODE_ID`. Do not flash two boards with the same ID - this value appears in every CSV row and is how the ML pipeline distinguishes nodes.

### LED Confirmation

| LED Pattern | Meaning |
|---|---|
| 3 quick blinks at startup | Board booted successfully |
| 1 blink every ~10 seconds | Transmitter is running and sending packets |
| No LED activity | Receiver is silently listening (normal) |

---

## Network Setup: Smartphone Hotspot + SSH

This setup lets your laptop control both Raspberry Pis wirelessly in the field with no router required.

### On your smartphone

Enable **Mobile Hotspot** and note the SSID and password or you can provide your ssid and password during RPi Setup.

### On each Raspberry Pi (do this at home before going to the field)

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

Add at the end of the file:

```
network={
    ssid="YourHotspotSSID"
    psk="YourHotspotPassword"
}
```

Save (`Ctrl+O`, `Enter`, `Ctrl+X`), then reboot:

```bash
sudo reboot
```

### Find the RPi IP addresses

Once both RPis are connected to the hotspot, find their IPs from your laptop:

```bash
# macOS / Linux
arp -a

# Or use nmap (replace subnet with your hotspot's, e.g. 192.168.43)
nmap -sn 192.168.43.0/24
```

Most smartphone hotspot screens also list connected devices with their IPs directly.

### SSH into each RPi

```bash
# Terminal 1 - RPi logging the first receiver
ssh pi@192.168.x.xxx

# Terminal 2 - RPi logging the second receiver
ssh pi@192.168.x.yyy
```

Default password is `raspberry` (change it with `passwd` after first login).

### Copy scripts to both RPis (one-time setup)

```bash
scp data_collection/logger.py    pi@192.168.x.xxx:~/
scp data_collection/verify.py    pi@192.168.x.xxx:~/

scp data_collection/logger.py    pi@192.168.x.yyy:~/
scp data_collection/verify.py    pi@192.168.x.yyy:~/
```

### Install the only Python dependency (run on both RPis)

```bash
pip3 install pyserial
```

---

## Data Collection Workflow

### Before the first session at any site - Range Test

Run `range_test_logger.py` **once per site** to find the maximum usable transmission distance. This can be run from your laptop or one RPi - only one receiver board is needed.

Connect one receiver node via USB and run:

```bash
python3 range_test_logger.py
```

You will be prompted:

```
Which node is the RECEIVER (connected to this device)? (A / B / C): B
Which node is the TRANSMITTER (powered by power bank)?  (A / B / C): A
```

The script tests 10 distances (10 m to 120 m, 2 minutes each) and prints a results table. The final line shows the **maximum usable distance**. Deploy all nodes at that separation for all subsequent main sessions at that site.

---

### Main Experiment - Step by Step

#### 1. Physical deployment in the field

- Place the **transmitter node** (connected to power bank) at one end of the site.
- Connect the two **receiver nodes** to their respective RPis via USB.
- Position both RPis at the measured maximum distance from the transmitter.
- Confirm your laptop and both RPis are connected to the smartphone hotspot.

#### 2. Check port assignments on each RPi

```bash
ls /dev/ttyACM*
# Expected: /dev/ttyACM0  (one board per RPi in this setup)
```

If `No such file or directory`, the board is not recognised - try a different USB cable.

#### 3. Start loggers on both RPis simultaneously

Open two SSH terminals. Start them as close in time as possible.

**SSH Terminal 1:**
```bash
python3 logger.py
```

**SSH Terminal 2:**
```bash
python3 logger.py
```

Both will prompt you interactively. Example for Session 1 (Node A transmitting):

```
Which node is the TRANSMITTER today? (A / B / C): A
  Transmitter : Node A  (powered by power bank, NOT connected to RPi)
  Receivers   : Node B and Node C  (connected to RPi via USB)

Available ports:
  [0]  /dev/ttyACM0   Adafruit Feather nRF52840

HINT: Plug Node B in FIRST, then Node C.
      On RPi: first board -> /dev/ttyACM0, second -> /dev/ttyACM1

Select port for Node B [0-0]: 0
  Node B -> /dev/ttyACM0
Select port for Node C [0-0]: 0

Environment options: bridge | garden | forest | river | lake
Enter environment name: bridge

Experiment type options: range_test | main
Enter experiment type: main
Duration: 30 minutes
```

> **Important:** Use the **exact same environment name** on both RPis. This string is embedded in the output filename and is how the ML pipeline groups files by environment.

After the confirmation summary, press **ENTER** on both terminals to start.

Each logger runs for exactly **30 minutes** and saves:
```
~/data/node<X>_<env>_main_<timestamp>.csv
```

#### 4. Power on the transmitter

Once both loggers show "Logging started", power on the transmitter node (connect to the power bank). Its LED blinks 3 times on boot, then once every ~10 seconds to confirm it is transmitting.

#### 5. Monitor live progress

Both terminals print a live status line every 10 seconds:
```
[STATUS] Elapsed: 5.0 min | Remaining: 25.0 min | Node B: 3012 pkts | Node C: 2998 pkts
```

A healthy session produces ~600 packets per minute per node (~10 pkt/s x 60 s).

#### 6. Verify the data immediately after each session

`verify.py` works on the RPi, your laptop (Mac/Windows/Linux), or anywhere with Python.

```bash
# Check all CSVs in the default folder (~/data/ on RPi, ~/Desktop/IoT_Data/ on Mac/Windows)
python3 verify.py

# Check a specific file
python3 verify.py nodeB_bridge_main_20260310_143022.csv

# Check all CSVs in a specific folder
python3 verify.py /path/to/folder/
```

Example output per file:

```
  ====================================================
  File     : nodeB_bridge_main_20260310_143022.csv
  Status   : OK
  ----------------------------------------------------
  Node IDs : 2
  Packets  : 18000
  Duration : 1800.0s  (30.0 min)
  Pkt rate : 10.0 pkt/s  (target >= 10)
  Complete : 100.0% of expected 30-min data
  RSSI     : avg=-71.4  min=-82  max=-61  (dBm)
  Gaps>250ms: 12  (0.1% loss)
  ====================================================
```

**Minimum acceptable quality per file:**

| Metric | Threshold |
|---|---|
| Packet rate | >= 8.0 pkt/s |
| Packet loss | < 5% |
| RSSI average | > -90 dBm |

If any file shows `WARNING`, **repeat that session** before moving on.

#### 7. Repeat across all environments and all transmitter roles

Run the same 30-minute session at all 5 environments:
`bridge` | `garden` | `forest` | `river` | `lake`

Then swap the transmitter to the next node and repeat. The logger will ask which node is transmitting each time - no code changes needed.

At the end you will have 30 CSV files across `~/data/` on both RPis:

```
nodeA_bridge_main_*.csv    nodeA_garden_main_*.csv    ... (when B or C transmitted)
nodeB_bridge_main_*.csv    nodeB_garden_main_*.csv    ... (when A or C transmitted)
nodeC_bridge_main_*.csv    nodeC_garden_main_*.csv    ... (when A or B transmitted)
```

---

## Copying Data Off the RPis

```bash
scp -r pi@192.168.x.xxx:~/data/ ./collected_data/rpi1/
scp -r pi@192.168.x.yyy:~/data/ ./collected_data/rpi2/
```

---

## CSV File Format

Every logged file uses this header:

```
node_id, pi_timestamp, board_timestamp_ms, rssi, lqi
```

| Column | Type | Description |
|---|---|---|
| `node_id` | int | 1 = Node A, 2 = Node B, 3 = Node C |
| `pi_timestamp` | float | Unix timestamp recorded by the Raspberry Pi (seconds) |
| `board_timestamp_ms` | int | `millis()` timestamp from the receiver board |
| `rssi` | int | Received Signal Strength Indicator (dBm, typically -40 to -100) |
| `lqi` | int | Link Quality Indicator (0-255, derived from RSSI by the firmware) |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `No serial ports found` | Charge-only USB cable | Replace with a data cable |
| Packet rate < 8 pkt/s | Nodes too far apart | Move closer; re-run range test |
| `/dev/ttyACM*` not found | Board not enumerated | Replug USB; run `sudo usermod -a -G dialout pi` then reboot |
| SSH timeout in the field | RPi dropped hotspot | Reboot RPi while the hotspot is active |
| RSSI avg < -90 dBm | Distance too large or obstruction | Reduce node separation |
| `logger.py` exits immediately | Port already in use | Close Arduino IDE Serial Monitor if open |
| Both nodes logging 0 packets | Transmitter not powered | Check power bank is on and LED is blinking |

---

