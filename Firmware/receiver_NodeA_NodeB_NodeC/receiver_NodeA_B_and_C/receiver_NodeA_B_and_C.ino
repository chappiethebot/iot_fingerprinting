/*
 * ============================================================
 *  IoT Fingerprinting Project — RECEIVER NODE (A, B, or C)
 *  Hardware : Adafruit Feather Bluefruit Sense (nRF52840)
 *  Protocol : IEEE 802.15.4 @ 2425 MHz (Channel 15)
 *
 *  Output via USB Serial (CSV format):
 *    node_id, board_timestamp_ms, rssi, lqi
 * ============================================================
 *  BEFORE FLASHING:
 *    - Set NODE_ID = 1 for Node A
 *    - Set NODE_ID = 2 for Node B
 *    - Set NODE_ID = 3 for Node C
 *
 *  Flash each board separately with its own correct NODE_ID.
 *  This value is written into every CSV row and is how the
 *  ML pipeline distinguishes which node received each packet.
 * ============================================================
 */

#include <Adafruit_TinyUSB.h>

#define NODE_ID       2         // ← SET TO 1 (Node A), 2 (Node B), OR 3 (Node C)
#define CHANNEL_FREQ  25        // Must match transmitter exactly
#define RX_TIMEOUT_MS 500       // How long to wait for a packet

static uint8_t rx_buf[128];

void radio_init() {
  NRF_RADIO->POWER = 1;

  NRF_RADIO->MODE = (15UL << RADIO_MODE_MODE_Pos);
  NRF_RADIO->FREQUENCY = CHANNEL_FREQ;

  NRF_RADIO->PCNF0 =
    (8UL  << RADIO_PCNF0_LFLEN_Pos)  |
    (0UL  << RADIO_PCNF0_S0LEN_Pos)  |
    (0UL  << RADIO_PCNF0_S1LEN_Pos)  |
    (2UL  << RADIO_PCNF0_PLEN_Pos)   |   // 802.15.4 zero preamble
    (1UL  << RADIO_PCNF0_CRCINC_Pos);

  NRF_RADIO->PCNF1 =
    (127UL << RADIO_PCNF1_MAXLEN_Pos)  |
    (0UL   << RADIO_PCNF1_STATLEN_Pos) |
    (0UL   << RADIO_PCNF1_BALEN_Pos)   |
    (0UL   << RADIO_PCNF1_ENDIAN_Pos)  |
    (0UL   << RADIO_PCNF1_WHITEEN_Pos);

  NRF_RADIO->CRCCNF  = (2UL << RADIO_CRCCNF_LEN_Pos) |
                        (2UL << RADIO_CRCCNF_SKIPADDR_Pos);
  NRF_RADIO->CRCPOLY = 0x011021UL;
  NRF_RADIO->CRCINIT = 0x0000UL;

  NRF_RADIO->SFD = 0xA7;  // 802.15.4 standard SFD

  NRF_RADIO->PACKETPTR = (uint32_t)rx_buf;
}

bool radio_receive(int8_t *rssi_out, uint8_t *lqi_out) {
  // Enable RX and wait for ramp-up
  NRF_RADIO->EVENTS_READY = 0;
  NRF_RADIO->TASKS_RXEN   = 1;
  while (!NRF_RADIO->EVENTS_READY);

  // Start listening
  NRF_RADIO->EVENTS_FRAMESTART = 0;
  NRF_RADIO->EVENTS_END        = 0;
  NRF_RADIO->TASKS_START       = 1;

  // ── Step 1: Wait for FRAMESTART (SFD detected = real packet arriving) ──
  // RSSI is only measured when an actual packet is being received,
  // not during background noise.
  uint32_t t_start = millis();
  while (!NRF_RADIO->EVENTS_FRAMESTART) {
    if ((millis() - t_start) > RX_TIMEOUT_MS) {
      // No packet arrived — disable and return
      NRF_RADIO->EVENTS_DISABLED = 0;
      NRF_RADIO->TASKS_DISABLE   = 1;
      while (!NRF_RADIO->EVENTS_DISABLED);
      return false;
    }
  }

  // ── Step 2: SFD detected — measure RSSI NOW (signal is present) ────────
  NRF_RADIO->EVENTS_RSSIEND  = 0;
  NRF_RADIO->TASKS_RSSISTART = 1;
  while (!NRF_RADIO->EVENTS_RSSIEND);  // takes ~0.25ms

  // RSSISAMPLE is positive: actual RSSI = -RSSISAMPLE dBm
  int8_t rssi = -(int8_t)(NRF_RADIO->RSSISAMPLE);

  // ── Step 3: Wait for packet to finish receiving ─────────────────────────
  while (!NRF_RADIO->EVENTS_END);

  // Check hardware CRC
  bool crc_ok = (NRF_RADIO->CRCSTATUS == 1);

  // Disable radio
  NRF_RADIO->EVENTS_DISABLED = 0;
  NRF_RADIO->TASKS_DISABLE   = 1;
  while (!NRF_RADIO->EVENTS_DISABLED);

  if (!crc_ok) return false;

  // Map RSSI [-100, -40] → LQI [0, 255]
  int lqi_raw = (int)(((float)(rssi + 100) / 60.0f) * 255.0f);
  if (lqi_raw < 0)   lqi_raw = 0;
  if (lqi_raw > 255) lqi_raw = 255;

  *rssi_out = rssi;
  *lqi_out  = (uint8_t)lqi_raw;
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  radio_init();

  // Print CSV header
  Serial.println("node_id,board_timestamp_ms,rssi,lqi");
}

void loop() {
  int8_t  rssi;
  uint8_t lqi;

  if (radio_receive(&rssi, &lqi)) {
    Serial.print(NODE_ID);
    Serial.print(",");
    Serial.print(millis());
    Serial.print(",");
    Serial.print(rssi);
    Serial.print(",");
    Serial.println(lqi);
  }
}
