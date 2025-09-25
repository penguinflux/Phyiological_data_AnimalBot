# AnimalBot - A robot that thinks it's an animal
This project is about a little robot car that I'm trying to make mimic the basic survival instincts of an animal. It runs on a Raspberry Pi and an Arduino, with some AI to interact with people. The whole idea is to see if it can survive by convincing people to help it out.

## What does it do?
The robot's "brain" (animal_brain.py) is a state machine that controls these main behaviors:

## 1. Hunt (for food)
The robot uses its camera to find people. When it sees a face, it will drive over, tilt its head up to look at the person, and ask you to "feed" it by pressing a button on its back. It also learns which sentences work best.

## 2. Escape (from predator)
I am its designated "hunter". Using face recognition, it only recognizes my face and knows it has to run away as soon as it sees me. If it spots me, it'll back up while playing a distress sound.

## 3. Reproduce (pass on its legacy)
If you "feed" it, it might ask you for one last favor: to help it "reproduce". It will ask you to scan a QR code that links to this GitHub repo, so you can build a copy of it.

## 4. Learn (practice makes perfect)
Every time it talks to someone, the robot adjusts the score of the sentence it just used based on the person's reaction (affirmative or negative). This way, its skill at asking for help gets better over time.

## Tech Specs
Here's everything inside the robot:

Brain: Raspberry Pi (4B or newer recommended)

Spinal Cord: Arduino (handles real-time stuff like motors and buttons)

Body: A standard robot car chassis with two continuous rotation servos for wheels.

Eyes & Neck: A Raspberry Pi camera mounted on a servo so it can tilt up and down.

Mouth & Ears: Custom NB3 I2S audio boards.

Touch: A physical button connected to the Arduino.

Core Logic: Python 3.

Vision: OpenCV (using a DNN model to find faces, and LBPH to recognize me).

Camera Driver: picamera2 library.

Speech Recognition: SpeechRecognition library.

Intent Analysis: Google Cloud Vertex AI (Gemini), for a low-latency cloud brain to figure out if you said "yes" or "no".

Hardware Driver: Custom NB3.Sound Python library.

## How to Get it Running
Just follow these steps on your Raspberry Pi.

## Step 1: Install System Dependencies
First, get the system ready. Run an update, then install the tools needed for the camera, audio, and other libraries.

sudo apt-get update
sudo apt-get install -y python3-libcamera flac espeak-ng ffmpeg

## Step 2: Set up the Python Environment
It's best to use a virtual environment so you don't mess up other Python projects on your system.

Go to the repo root directory
cd ~/NoBlackBoxes/LastBlackBox/

Create a new virtual environment
python3 -m venv _tmp/LBB

Activate it
source _tmp/LBB/bin/activate

## Step 3: Install Python Packages
With the environment active, install all the Python dependencies with one command. This assumes you have the requirements.txt file in the Survival_bot folder.

Go to the project folder
cd Survival_bot/

Install everything
pip install -r requirements.txt

## Step 4: Configure Google Cloud (Vertex AI)
This part lets the robot connect to the internet to "think". You'll need to set up your Google Cloud project and get authorization. I wrote down the detailed steps in another file: Vertex AI Config Guide.

## Step 5: Teach it to Recognize You
You need to let the robot know who the "hunter" is.

Make sure the HUNTER_FACE_ID variable at the top of animal_brain.py is set to the name you want to use (e.g., "hunter_John").

Run the training script:

python trainer.py

It will ask for a name - type in the exact same ID you set in the script. Then let it take 30 pictures of your face. It will then generate the trainer.yml and names.json memory files.

# Run
After you've done all the steps above, make sure your virtual environment is active (LBB), then just run the main script:

python animal_brain.py
