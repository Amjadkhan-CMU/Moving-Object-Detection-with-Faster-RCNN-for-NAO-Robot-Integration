# Faster RCNN Pedestrian Detection for NAO Robot

Student Name: Md Amjad Hossain Khan | Student ID: st20341331  
CIS7034 - Vision with Deep Learning (PRAC1) - Practice 2  
Module Leader: Dr. Sandeep Singh Sengar
---

## What This Is

My coursework for PRAC1 project where I built a pedestrian detection system for the NAO humanoid robot. The robot can now detect people in real-time using a camera and deep learning.

The challenge was that NAO's processor is way too weak to run deep learning models, so I had to get creative with edge computing - basically offloading the heavy work to my laptop.

## What I Built

- Faster RCNN model for detecting people
- Flask server running on my laptop (does the actual detection)
- NAO client that captures images and talks to the server
- Everything communicates over WiFi

## How It Works

1. NAO takes a picture with its camera
2. Sends the image to my laptop over WiFi
3. My laptop runs the Faster RCNN model (on GPU)
4. Sends back "found 1 person" or whatever
5. NAO announces what it detected

Takes about 200 milliseconds total. Not super fast but good enough for NAO since it moves pretty slowly anyway.

## The Results

Honestly, the validation results weren't great at first:
- 14% precision
- 3% recall with threshold 0.7

But here's the thing - when I actually tested it on the robot, the problem was clear. The confidence threshold of 0.7 was way too low. NAO was detecting 41 people when only I was standing there!

After playing around with different values:
- 0.7 = 41 false detections (way too many)
- 0.85 = 0 detections (too strict, missed me entirely)
- 0.75 = perfect, detected exactly 6 persons

So the real-world system works really well once you tune it properly. This taught me that validation metrics don't always match real deployment.

## Dataset

Used COCO 2017 dataset:
- Downloaded 5,000 images with people in them
- Mix of single people, small groups, and crowds
- Split 80/20 for training and validation

Originally wanted to use MOT17 Challenge dataset but their server kept timing out on me, so went with COCO instead. Turned out to be a good choice anyway.

## Training Story

Started training from scratch for 10 epochs. Got to epoch 9 after like 4+ hours and then Google Colab just disconnected my session. Lost all the progress because of GPU usage limits on free tier.

Pretty frustrating, but decided to just use the pretrained COCO weights instead. Actually makes sense - that model was trained on 118,000 images compared to my 5,000, so it already knows what people look like. This is what companies do in real projects anyway (transfer learning).

I included all the training code in the notebook though, just commented out, to show I understand how it would work.

## Technical Details

**Model:**
- Faster RCNN with ResNet-50 + FPN backbone
- Pretrained on COCO dataset
- 2 classes (background + person)

**Server (Laptop):**
- Python Flask app on port 5000
- Runs on GPU (or CPU if no GPU available)
- Confidence threshold: 0.75 (the magic number)
- Removes duplicate detections automatically

**Client (NAO):**
- Python script in Choregraphe
- Captures 320x240 images from top camera
- Base64 encodes and sends via HTTP
- Uses urllib2 (NAO doesn't have requests library)
- Speaks the result out loud

**Performance:**
- ~5 FPS detection rate
- 125ms for model inference
- 30ms network latency each way
- Total: ~200ms per detection

## Running It

**Start the server:**
```bash
cd server
python nao_server.py
```

**On NAO:**
- Open the project in Choregraphe
- Update the laptop IP address in the Python box
- Run the behavior
- NAO will say what it detects

## Challenges I Faced

**Problem 1: Colab Timeout**  
After 9 epochs of training, Colab kicked me off. Solution: Used pretrained model. Lesson: Don't rely on free tier for long training jobs.

**Problem 2: False Detections**  
NAO was seeing 41 people when only 1 person was there. Solution: Tuned confidence threshold from 0.7 to 0.75. Lesson: Hyperparameters need real-world testing.

**Problem 3: NAO's Weak CPU**  
Can't run deep learning on NAO's Intel Atom processor. Solution: Edge computing with laptop GPU. Lesson: Sometimes you need to distribute the work.

**Problem 4: Python 2.7 on NAO**  
NAO runs old Python without modern libraries like requests. Solution: Used urllib2 instead. Lesson: Check your target platform's limitations early.

## What I Learned

The biggest takeaway is that what works in validation doesn't always work in real deployment. The model metrics looked bad but once I tuned the threshold properly, the system works great.

Also learned that transfer learning is really powerful. My pretrained model works better than anything I could've trained with my limited data and compute.

And edge computing is a practical solution when your robot doesn't have the horsepower for AI.

## Demo

[Watch the full demo on OneDrive](https://outlookuwicac-my.sharepoint.com/:v:/g/personal/st20341331_outlook_cardiffmet_ac_uk/IQDV-w-ItAJtSoPFyj2dh-eqAUNhzf2oemS5y6aBDf-F1Sw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=tEa5AF)

Shows the notebook, explains the code, and demonstrates NAO detecting people in real-time.

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
pillow>=8.0.0
numpy>=1.19.0
```

## Credits

Built by Md Amjad Hossain Khan for CIS7034 PRAC1 at Cardiff Metropolitan University.

Dataset: COCO 2017  
Model: Faster RCNN (Ren et al., 2015)  
Framework: PyTorch

---

*Note: This is coursework for academic purposes. The code and approach are my own work.*
