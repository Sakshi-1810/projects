import os
import cv2
from ultralytics import YOLO
from flask import Flask, request, send_file, render_template
import logging

app = Flask(__name__)

INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'yolov8m.pt'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

try:
    model = YOLO(MODEL_PATH)
    class_names = getattr(model, 'names', {i: f'class_{i}' for i in range(1000)})
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    exit(1)

def draw_boxes(frame, results):
    """Draw bounding boxes and labels on frame"""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def process_image(image_path):
    """Process single image file"""
    try:
        result = model(image_path)[0]
        image = cv2.imread(image_path)
        processed = draw_boxes(image, result)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
        cv2.imwrite(output_path, processed)
        return output_path
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return None

def process_video(video_path):
    """Process video file frame by frame"""
    try: 
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use H.264 (avc1) for MP4 compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        # Ensure correct file extension
        base_name, ext = os.path.splitext(os.path.basename(video_path))
        output_basename = f"processed_{base_name}{ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_basename)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            processed_frame = draw_boxes(frame, results)
            out.write(processed_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if frame_count == 0:
            logging.error("No frames processed for video")
            return None
        
        return output_path
    except Exception as e:
        logging.error(f"Video processing error: {e}")
        return None
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            os.makedirs(INPUT_FOLDER, exist_ok=True)
            input_path = os.path.join(INPUT_FOLDER, file.filename)
            file.save(input_path)
            
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                result_path = process_image(input_path)
            elif file.filename.lower().endswith(('.mp4', '.avi')):
                result_path = process_video(input_path)
            else:
                return "Unsupported file type", 400
                
            if result_path:
                return render_template(
                    'index.html',
                    original_file=file.filename,
                    result_file=os.path.basename(result_path)
                )
            return "Processing failed", 500
            
    return render_template('index.html')


@app.route('/input/<filename>')
def input_file(filename):
    return send_file(os.path.join(INPUT_FOLDER, filename))

@app.route('/output/<filename>')
def output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True) 