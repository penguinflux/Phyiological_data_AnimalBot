/*
  Arduino Motor Controller (V4 - Continuous Rotation Servo Edition)
  File: motor_controller_v4_continuous_servo.ino

  描述:
  这个程序是为使用两个“连续旋转伺服电机”作为驱动轮的机器人定制的。
  它将来自树莓派的高级指令（如'f'）转换为对伺服电机的特定角度写入（如90, 0, 180）。
  同时，它集成了摄像头俯仰舵机和喂食按钮的功能。

  --- 硬件连接 ---
  - 右轮连续伺服电机 -> Arduino D9
  - 左轮连续伺服电机 -> Arduino D10
  - 摄像头俯仰舵机   -> Arduino D11
  - 喂食按钮         -> Arduino D2 (另一端接GND)

  --- 串口指令 ---
  - 'f': 前进
  - 'b': 后退
  - 'l': 左转
  - 'r': 右转
  - 's': 停止
  - 'u': 摄像头抬头
  - 'd': 摄像头低头
  - 'p<angle>': 设置摄像头到指定角度 (例如: 'p90')
  - (发送到树莓派) 'k': 喂食按钮被按下
*/

#include <Servo.h>

// --- 定义引脚 ---
const int RIGHT_MOTOR_PIN = 9;
const int LEFT_MOTOR_PIN = 10;
const int CAMERA_SERVO_PIN = 3;
const int FEED_BUTTON_PIN = 4;

// --- 创建伺服电机对象 ---
Servo right_motor;
Servo left_motor;
Servo camera_servo;

// --- 配置参数 ---
const int MOTOR_SPEED = 60; // 速度参数，可调 (0-90)。越大越快。

// 摄像头舵机变量
int camera_servo_angle = 90;
const int ANGLE_STEP = 3;
const int MIN_ANGLE = 20;
const int MAX_ANGLE = 160;

// 按钮状态变量
int lastButtonState = HIGH;

void setup() {
  Serial.begin(115200);

  // 连接伺服电机
  right_motor.attach(RIGHT_MOTOR_PIN);
  left_motor.attach(LEFT_MOTOR_PIN);
  camera_servo.attach(CAMERA_SERVO_PIN);

  // 初始化按钮引脚 (使用内置上拉电阻)
  pinMode(FEED_BUTTON_PIN, INPUT_PULLUP);

  // 机器人上电后停止并让摄像头归中
  stopMotors();
  camera_servo.write(camera_servo_angle);
  delay(500);
}

void loop() {
  // 1. 检查来自树莓派的指令
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'p') {
      int angle = Serial.parseInt();
      setCameraAngle(angle);
    } else {
      switch (command) {
        case 'f': moveForward(); break;
        case 'b': moveBackward(); break;
        case 'l': turnLeft(); break;
        case 'r': turnRight(); break;
        case 's': stopMotors(); break;
        case 'u': camera_servo_angle += ANGLE_STEP; setCameraAngle(camera_servo_angle); break;
        case 'd': camera_servo_angle -= ANGLE_STEP; setCameraAngle(camera_servo_angle); break;
      }
    }
  }

  // 2. 检查按钮是否被按下
  checkFeedButton();

  delay(10); // 短暂延时以稳定系统
}

// --- 动作函数定义 (已适配连续旋转伺服) ---

void moveForward() {
  // 一个正转，一个反转，取决于您的电机安装方向
  // 90 + speed = 一个方向全速, 90 - speed = 另一个方向全速
  left_motor.write(90 + MOTOR_SPEED);
  right_motor.write(90 - MOTOR_SPEED);
}

void moveBackward() {
  left_motor.write(90 - MOTOR_SPEED);
  right_motor.write(90 + MOTOR_SPEED);
}

void turnLeft() {
  // 原地左转：左轮后退，右轮前进
  left_motor.write(90 - MOTOR_SPEED);
  right_motor.write(90 - MOTOR_SPEED);
}

void turnRight() {
  // 原地右转：左轮前进，右轮后退
  left_motor.write(90 + MOTOR_SPEED);
  right_motor.write(90 + MOTOR_SPEED);
}

void stopMotors() {
  // 90度是连续旋转伺服的停止点
  left_motor.write(90);
  right_motor.write(90);
}

void setCameraAngle(int angle) {
  camera_servo_angle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
  camera_servo.write(camera_servo_angle);
}

void checkFeedButton() {
  int buttonState = digitalRead(FEED_BUTTON_PIN);
  
  // 检查按钮状态是否发生变化（从“未按”到“按下”）
  if (buttonState != lastButtonState) {
    if (buttonState == LOW) { // LOW表示按钮被按下 (因为我们用了INPUT_PULLUP)
      Serial.println('k'); // 向树莓派发送'k'信号
    }
    delay(50); // 简单的延时消抖
  }
  lastButtonState = buttonState;
}