#include <Wire.h>
#include <math.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

const int MPU_ADDR = 0x68;

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
volatile bool deviceConnected = false;

struct __attribute__((packed)) Sample {
  uint16_t accel_mag;
  uint16_t gyro_mag;
};

struct __attribute__((packed)) TremorPacket {
  uint16_t seq;
  Sample samples[4];
  // Removed the 2-byte padding so the packet is exactly 18 bytes
};

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      BLEDevice::startAdvertising();
    }
};

void writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

void sensorSamplingTask(void *pvParameters) {
  const TickType_t xFrequency = 10 / portTICK_PERIOD_MS;
  TickType_t xLastWakeTime = xTaskGetTickCount();

  TremorPacket txPacket;
  txPacket.seq = 0;
  uint8_t sampleIndex = 0;

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    // Read 14 bytes from MPU6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)MPU_ADDR, (uint8_t)14, (uint8_t)true);

    if (Wire.available() >= 14) {
      int16_t ax = (Wire.read() << 8) | Wire.read();
      int16_t ay = (Wire.read() << 8) | Wire.read();
      int16_t az = (Wire.read() << 8) | Wire.read();
      Wire.read(); Wire.read(); // Skip temperature bytes
      int16_t gx = (Wire.read() << 8) | Wire.read();
      int16_t gy = (Wire.read() << 8) | Wire.read();
      int16_t gz = (Wire.read() << 8) | Wire.read();

      float raw_accel = sqrt((float)ax * ax + (float)ay * ay + (float)az * az);
      txPacket.samples[sampleIndex].accel_mag = (uint16_t)round((raw_accel * 1000.0f) / 16384.0f);

      float raw_gyro = sqrt((float)gx * gx + (float)gy * gy + (float)gz * gz);
      txPacket.samples[sampleIndex].gyro_mag = (uint16_t)round((raw_gyro * 10.0f) / 131.0f);

      sampleIndex++;

      // ONLY notify when 4 samples are gathered (exactly every 40ms)
      if (sampleIndex >= 4) {
        if (deviceConnected) {
          pCharacteristic->setValue((uint8_t*)&txPacket, sizeof(txPacket));
          pCharacteristic->notify();
        }
        txPacket.seq += 4; 
        sampleIndex = 0;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400kHz fast I2C to minimize bus read time

  // Configure MPU6050
  writeRegister(0x6B, 0x00); // Wake up
  writeRegister(0x1C, 0x00); // +/- 2g
  writeRegister(0x1B, 0x00); // +/- 250 deg/s

  // Initialize BLE
  BLEDevice::init("TremorBand");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_NOTIFY
                    );
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();

  // Start 100 Hz sampling task pinned to Core 1
  xTaskCreatePinnedToCore(
    sensorSamplingTask,
    "SensorTask",
    4096,
    NULL,
    1,
    NULL,
    1
  );
}

void loop() {
  // Main loop remains clear of slow Serial.prints to protect timing
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}
