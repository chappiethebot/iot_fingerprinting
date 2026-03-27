/*
 * ============================================================
 *  IoT Fingerprinting Project — NODE A (TRANSMITTER)
 *  Hardware : Adafruit Feather Bluefruit Sense (nRF52840)
 *  Protocol : IEEE 802.15.4 @ 2425 MHz (Channel 15)
 *
 *  LED blinks 3 times on startup to confirm boot.
 *  LED blinks once every ~10 seconds while transmitting.
 * ============================================================
 *  FLASH THIS TO NODE A OR B OR C.
 *  Power it from a LiPo battery in the field.
 * ============================================================
 */

#include <Adafruit_TinyUSB.h>   // ← fixes Serial/USB compile errors

#define TX_INTERVAL_MS  100     // 100ms = 10 packets per second
#define CHANNEL_FREQ    25      // 2400 + 25 = 2425 MHz (802.15.4 ch 15)
#define TX_POWER        8       // +8 dBm = maximum power
#define LED_PIN         LED_BUILTIN

static uint8_t  tx_buf[128];
static uint32_t seq_num = 0;

void radio_init() {
  NRF_RADIO->POWER = 1;

  NRF_RADIO->MODE = (15UL << RADIO_MODE_MODE_Pos);
  NRF_RADIO->FREQUENCY = CHANNEL_FREQ;
  NRF_RADIO->TXPOWER = (TX_POWER << RADIO_TXPOWER_TXPOWER_Pos);

  NRF_RADIO->PCNF0 =
    (8UL  << RADIO_PCNF0_LFLEN_Pos)  |
    (0UL  << RADIO_PCNF0_S0LEN_Pos)  |
    (0UL  << RADIO_PCNF0_S1LEN_Pos)  |
    (2UL  << RADIO_PCNF0_PLEN_Pos)   |
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

  NRF_RADIO->SFD = 0xA7;
}

void radio_send(uint8_t *payload, uint8_t payload_len) {
  tx_buf[0] = payload_len + 2;
  memcpy(&tx_buf[1], payload, payload_len);

  NRF_RADIO->PACKETPTR = (uint32_t)tx_buf;

  NRF_RADIO->EVENTS_READY = 0;
  NRF_RADIO->TASKS_TXEN   = 1;
  while (!NRF_RADIO->EVENTS_READY);

  NRF_RADIO->EVENTS_END  = 0;
  NRF_RADIO->TASKS_START = 1;
  while (!NRF_RADIO->EVENTS_END);

  NRF_RADIO->EVENTS_DISABLED = 0;
  NRF_RADIO->TASKS_DISABLE   = 1;
  while (!NRF_RADIO->EVENTS_DISABLED);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  radio_init();

  // 3 quick blinks = board started successfully
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(150);
    digitalWrite(LED_PIN, LOW);  delay(150);
  }
}

void loop() {
  uint8_t  payload[9];
  uint32_t ts = millis();

  payload[0] = 0x01;  // Node A = 1
  payload[1] = (seq_num >> 24) & 0xFF;
  payload[2] = (seq_num >> 16) & 0xFF;
  payload[3] = (seq_num >>  8) & 0xFF;
  payload[4] = (seq_num      ) & 0xFF;
  payload[5] = (ts >> 24) & 0xFF;
  payload[6] = (ts >> 16) & 0xFF;
  payload[7] = (ts >>  8) & 0xFF;
  payload[8] = (ts      ) & 0xFF;

  radio_send(payload, 9);
  seq_num++;

  // Blink LED every 100 packets (~10 sec) — field confirmation
  if (seq_num % 100 == 0) {
    digitalWrite(LED_PIN, HIGH); delay(50);
    digitalWrite(LED_PIN, LOW);
  }

  delay(TX_INTERVAL_MS);
}
