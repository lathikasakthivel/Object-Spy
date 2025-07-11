# ObjectSpy — YOLOv4 Object Detection Web App with Real-Time Camera Feed  

ObjectSpy is a lightweight, interactive web application that uses **YOLOv4 (You Only Look Once)** for detecting multiple objects in both:
- **Uploaded images**
- **Live webcam video feed**

Built with **Python**, **Flask**, and **OpenCV**, the app detects objects, draws bounding boxes with labels and confidence scores, counts total detections, and displays processed results via a clean, animated browser interface.

---

## Project Objective  

To create an intuitive, browser-based object detection tool for both static images and real-time webcam streams, demonstrating practical use of YOLOv4 deep learning in computer vision.

---

## What Problem Does It Solve?  

Most object detection systems require command-line operation or heavy software setups.  
**ObjectSpy** offers a clean, no-install web experience where:
- Anyone can upload an image or access their webcam.
- Instantly detect multiple objects.
- View processed images or video with labeled bounding boxes and object counts.

---

## Features  

✅ Upload image and detect multiple objects  
✅ Count total detected objects  
✅ Draw bounding boxes and labels with confidence scores  
✅ Real-time webcam object detection via browser stream  
✅ Responsive, modern animated UI  
✅ Detailed detection result page  

---

## Tech Stack  

| Component        | Technology              |
|:----------------|:------------------------|
| **Backend**      | Python, Flask            |
| **Object Detection** | YOLOv4 (Darknet via OpenCV DNN) |
| **Frontend**     | HTML, CSS , JavaScript |
| **Deployment**   | Flask dev server |

---

## How It Works  

### Image Detection
1. **User uploads an image** via browser.
2. Flask saves it locally.
3. YOLOv4 processes the image:
   - Converts image to a blob.
   - Runs YOLO forward pass.
   - Detects objects with confidence > 0.5.
   - Draws bounding boxes and labels.
4. Image is base64 encoded and sent back.
5. Webpage displays:
   - Processed image.
   - Object count.
   - Object names and confidences.

---

### Real-Time Webcam Detection  
1. Browser accesses `/video_feed` route.
2. OpenCV captures webcam frames.
3. Each frame is processed by YOLOv4:
   - Detects objects.
   - Draws bounding boxes and labels.
4. Frame is encoded as JPEG.
5. Flask streams frames to browser in real-time via multipart MJPEG response.

---

## Example Outputs  

- Processed images/video with labels like:
- 
