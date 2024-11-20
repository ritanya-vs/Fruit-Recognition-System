
# Smart Fruit Recognition System

## About the Project

The **Smart Fruit Recognition System** uses an **ESP32-CAM module** integrated with **Arduino** and powered by a **TensorFlow Lite** model for real-time object detection. The system recognizes different fruits by analyzing images captured by the ESP32-CAM's camera. Upon successful recognition, the fruit's name is displayed on a connected interface via **Streamlit**.

### Key Features:
1. **Real-Time Object Detection**:  
   - The system uses a TensorFlow Lite model to detect fruits in images captured by the ESP32-CAM module.
  
2. **Interactive Feedback**:  
   - Once a fruit is detected, its name is displayed on a **Streamlit** interface, providing immediate feedback.

3. **User-Friendly Interface**:  
   - The interactive web interface allows users to view the last captured image and the detected fruit name.



---

## Hardware Specifications:
- **Espressif ESP32-CAM**: Captures images of the fruits.
- **Arduino UNO**: Controls peripherals and coordinates communication.

---

## Software Specifications:
- **Arduino IDE**: Used for programming the ESP32-CAM.
- **Programming Language**: Python for backend logic and data processing.
- **Libraries Used**:
  - **TensorFlow**: For fruit recognition with a pre-trained deep learning model.
  - **Streamlit**: For creating the user interface that displays predictions.
  
---

## How to Run the Project

### Prerequisites:
1. Install **Arduino IDE** and set up the **ESP32** library.
2. Install **Python 3.x** and the following libraries:
   ```bash
   pip install tensorflow streamlit 
   ```
3. Download dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

### Steps to Run:

1. **Upload Code to ESP32-CAM**:  
   - Open the `camerawebserver.ino` file in Arduino IDE, set up the appropriate network credentials, and upload the code to the ESP32-CAM.

2. **Train the TensorFlow Model**:  
   - Run the `fruit_cnn.py` script to train a CNN model for fruit classification. The model will be saved as `cnn_fruit_classification.tflite`.

3. **Run the Streamlit Interface**:  
   - Open the `app.py` file and run the following command to start the Streamlit app:
     ```bash
     streamlit run app.py
     ```

4. **Use the System**:
   - Once the system is running, the ESP32-CAM will capture images and send them to the TensorFlow Lite model for fruit detection.
   - The Streamlit interface will display the captured image along with the recognized fruit name.

---


