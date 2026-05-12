# NAO Robot Detection Server
# Md Amjad Hossain Khan - st20341331
# PRAC1 CIS7034_Computer Vision with Deep Learning
# This runs on my laptop and receives images from NAO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import base64

app = Flask(__name__)

print("Starting NAO Detection Server...")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# Load my trained weights
model.load_state_dict(torch.load("faster_rcnn.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")
print("Server ready - waiting for NAO...")

@app.route('/detect', methods=['POST'])
def detect_person():
    """
    This receives images from NAO and runs detection
    Returns number of people detected
    """
    try:
        print("\nReceived image from NAO")
        
        # Get the base64 encoded image
        data = request.get_json()
        image_b64 = data['image']
        
        # Decode it
        image_bytes = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # NAO sends images in different sizes, need to try each one
        print("Decoding image...")
        image = None
        possible_sizes = [
            (240, 320, 3),  # QVGA resolution
            (480, 640, 3),  # VGA resolution
            (120, 160, 3),  # QQVGA resolution
        ]
        
        for h, w, c in possible_sizes:
            try:
                img = image_array.reshape((h, w, c))
                # NAO uses BGR, need to convert to RGB
                img = img[:, :, ::-1]
                image = Image.fromarray(img, 'RGB')
                print(f"Image size: {w}x{h}")
                break
            except:
                continue
        
        if image is None:
            print("Failed to decode image")
            return jsonify({
                'person_detected': False,
                'num_people': 0,
                'status': 'error'
            })
        
        # Run the detection
        print("Running detection...")
        img_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        # Get the predictions
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        # Filter for people with good confidence
        # After testing, 0.75 works best - not too strict, not too loose
        confidence_threshold = 0.75
        
        detected_people = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            if label == 1 and score > confidence_threshold:
                detected_people.append({
                    'box': boxes[i].tolist(),
                    'score': float(score)
                })
        
        # Remove duplicate detections (when model detects same person multiple times)
        if len(detected_people) > 1:
            detected_people.sort(key=lambda x: x['score'], reverse=True)
            
            final_people = []
            for person in detected_people:
                # Check if this overlaps with already detected people
                is_duplicate = False
                for existing in final_people:
                    if boxes_overlap(person['box'], existing['box']):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    final_people.append(person)
            
            detected_people = final_people
        
        num_people = len(detected_people)
        
        print(f"Detected {num_people} people")
        for i, person in enumerate(detected_people):
            print(f"  Person {i+1}: {person['score']:.2f} confidence")
        
        return jsonify({
            'person_detected': num_people > 0,
            'num_people': num_people,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'person_detected': False,
            'num_people': 0,
            'status': 'error'
        })

def boxes_overlap(box1, box2):
    """Check if two boxes overlap too much (means same person detected twice)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Find intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    overlap = intersection / union if union > 0 else 0
    return overlap > 0.4  # More than 40% overlap = same person

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'running', 'device': str(device)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
