import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ---------- Load TFLite Model ----------
interpreter = tflite.Interpreter(model_path="/home/hello1/Desktop/New/APP_FIP_PROJECT/bottlefillmodel_v3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Open camera stream ----------
cap = cv2.VideoCapture("http://192.168.31.180:4747/video")

ret, frame = cap.read()
if not ret:
    print("Failed to capture frame")
    cap.release()
    exit()

# ---------- Predict function ----------
def predict_frame(frame):
    # Resize to the trained input size (128x128)
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)

# ---------- Run prediction ----------
predicted_class = predict_frame(frame)

# ---------- Class mapping ----------
class_mapping = {
    0: ("Empty", "Reject", (0, 0, 255)),
    1: ("Underfilled", "Reject", (0, 0, 255)),
    2: ("Underfilled", "Reject", (0, 0, 255)),
    3: ("Acceptable Fill", "Accept", (0, 255, 0)),
    4: ("Overflow", "Reject", (0, 0, 255))
}

status, decision, color = class_mapping.get(predicted_class, ("Unknown", "Reject", (255, 255, 255)))

print(f"Prediction: {status} → {decision}")

# ---------- Show the captured frame with label ----------
cv2.putText(frame, f"{status} ({decision})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
cv2.imshow("Bottle Fill Detection", frame)
cv2.waitKey(0)  # Wait until a key is pressed

# Cleanup
cap.release()
cv2.destroyAllWindows()

