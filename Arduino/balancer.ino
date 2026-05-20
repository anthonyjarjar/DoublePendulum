const int STEP_PIN = 2;
const int DIR_PIN  = 3;
const int EN_PIN   = 4;

const int ENC_A = 18;
const int ENC_B = 19;

const float COUNTS_PER_REV = 1024.0;

const int ENCODER_SIGN = +1;

volatile long encoderCount = 0;
volatile uint8_t lastAB = 0;

long zeroCount = 0;

const int8_t QUAD_TABLE[16] = {
  0, -1, +1,  0,
 +1,  0,  0, -1,
 -1,  0,  0, +1,
  0, +1, -1,  0
};

long targetStepRate = 0;
long currentStepRate = 0;

unsigned long lastStepMicros = 0;
unsigned long lastRampMicros = 0;

const long MAX_STEP_RATE = 9000;
const long MAX_ACCEL = 15000;
const int STEP_PULSE_US = 3;

float lastTheta = 0.0;
unsigned long lastThetaMicros = 0;

const unsigned long SEND_INTERVAL_MICROS = 10000;  // 100 Hz
unsigned long lastSendMicros = 0;

String inputString = "";

void updateEncoder() {
  uint8_t A = digitalRead(ENC_A);
  uint8_t B = digitalRead(ENC_B);

  uint8_t newAB = (A << 1) | B;
  uint8_t index = (lastAB << 2) | newAB;

  encoderCount += ENCODER_SIGN * QUAD_TABLE[index];

  lastAB = newAB;
}

float wrapAngle(float theta) {
  while (theta > PI) {
    theta -= 2.0 * PI;
  }

  while (theta < -PI) {
    theta += 2.0 * PI;
  }

  return theta;
}

float angleDifference(float thetaNew, float thetaOld) {
  float dtheta = thetaNew - thetaOld;

  while (dtheta > PI) {
    dtheta -= 2.0 * PI;
  }

  while (dtheta < -PI) {
    dtheta += 2.0 * PI;
  }

  return dtheta;
}

float countToTheta(long count, long zero) {
  long relativeCount = count - zero;
  float theta = relativeCount * 2.0 * PI / COUNTS_PER_REV;
  return wrapAngle(theta);
}

void setTargetStepRate(long newRate) {
  if (newRate > MAX_STEP_RATE) {
    newRate = MAX_STEP_RATE;
  }

  if (newRate < -MAX_STEP_RATE) {
    newRate = -MAX_STEP_RATE;
  }

  targetStepRate = newRate;
}

void zeroEncoder() {
  noInterrupts();
  zeroCount = encoderCount;
  interrupts();

  lastTheta = 0.0;
  lastThetaMicros = micros();

  targetStepRate = 0;
  currentStepRate = 0;
  digitalWrite(STEP_PIN, LOW);

  Serial.println("ZEROED");
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) {
    return;
  }

  if (cmd == "Z" || cmd == "z") {
    zeroEncoder();
    return;
  }

  if (cmd.startsWith("S:")) {
    long newRate = cmd.substring(2).toInt();
    setTargetStepRate(newRate);
    return;
  }

  long newRate = cmd.toInt();
  setTargetStepRate(newRate);
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      handleCommand(inputString);
      inputString = "";
    } else if (c != '\r') {
      inputString += c;
    }
  }
}

void updateRamp() {
  unsigned long now = micros();
  unsigned long dtMicros = now - lastRampMicros;

  if (dtMicros < 1000) {
    return;
  }

  lastRampMicros = now;

  float dt = dtMicros / 1000000.0;
  long maxChange = (long)(MAX_ACCEL * dt);

  if (maxChange < 1) {
    maxChange = 1;
  }

  if (currentStepRate < targetStepRate) {
    currentStepRate += maxChange;

    if (currentStepRate > targetStepRate) {
      currentStepRate = targetStepRate;
    }
  } else if (currentStepRate > targetStepRate) {
    currentStepRate -= maxChange;

    if (currentStepRate < targetStepRate) {
      currentStepRate = targetStepRate;
    }
  }
}

void generateSteps() {
  long rate = currentStepRate;

  if (rate == 0) {
    return;
  }

  if (rate > 0) {
    digitalWrite(DIR_PIN, HIGH);
  } else {
    digitalWrite(DIR_PIN, LOW);
    rate = -rate;
  }

  unsigned long now = micros();
  unsigned long stepIntervalMicros = 1000000UL / rate;

  if (now - lastStepMicros >= stepIntervalMicros) {
    lastStepMicros = now;

    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(STEP_PULSE_US);
    digitalWrite(STEP_PIN, LOW);
  }
}

void sendEncoderState() {
  unsigned long now = micros();

  if (now - lastSendMicros < SEND_INTERVAL_MICROS) {
    return;
  }

  lastSendMicros = now;

  noInterrupts();
  long count = encoderCount;
  long zero = zeroCount;
  interrupts();

  float theta = countToTheta(count, zero);

  float dt = 0.0;

  if (lastThetaMicros != 0) {
    dt = (now - lastThetaMicros) / 1000000.0;
  }

  float thetaDot = 0.0;

  if (dt > 0.0) {
    float dtheta = angleDifference(theta, lastTheta);
    thetaDot = dtheta / dt;
  }

  lastTheta = theta;
  lastThetaMicros = now;

  Serial.print("E,");
  Serial.print(theta, 6);
  Serial.print(",");
  Serial.print(thetaDot, 6);
  Serial.print(",");
  Serial.println(count);
}

void setup() {
  Serial.begin(115200);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  // DRV8825 enable is active LOW
  digitalWrite(EN_PIN, LOW);

  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);

  uint8_t A = digitalRead(ENC_A);
  uint8_t B = digitalRead(ENC_B);
  lastAB = (A << 1) | B;

  attachInterrupt(digitalPinToInterrupt(ENC_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), updateEncoder, CHANGE);

  inputString.reserve(32);

  lastThetaMicros = micros();
  lastRampMicros = micros();

  Serial.println("ARDUINO_READY");
  Serial.println("Send Z to zero. Send S:<rate> to command stepper.");
}

void loop() {
  readSerialCommands();
  updateRamp();
  generateSteps();
  sendEncoderState();
}