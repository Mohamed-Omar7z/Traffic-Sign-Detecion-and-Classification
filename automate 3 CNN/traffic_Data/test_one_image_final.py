import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model(r"D:\al_test\pythonXvs\Automated Traffic Sign Project\automate 3 CNN\traffic_Data\best_model.keras")

# CLASS NAMES - by actual ClassId (0-42)
class_names = {
    0:  'Speed limit 20',       1:  'Speed limit 30',
    2:  'Speed limit 50',       3:  'Speed limit 60',
    4:  'Speed limit 70',       5:  'Speed limit 80',
    6:  'End speed limit 80',   7:  'Speed limit 100',
    8:  'Speed limit 120',      9:  'No passing',
    10: 'No passing >3.5t',     11: 'Right of way',
    12: 'Priority road',        13: 'Yield',
    14: 'Stop',                 15: 'No vehicles',
    16: 'No vehicles >3.5t',    17: 'No entry',
    18: 'General caution',      19: 'Curve left',
    20: 'Curve right',          21: 'Double curve',
    22: 'Bumpy road',           23: 'Slippery road',
    24: 'Road narrows right',   25: 'Road work',
    26: 'Traffic signals',      27: 'Pedestrians',
    28: 'Children crossing',    29: 'Bicycles crossing',
    30: 'Ice/Snow',             31: 'Wild animals',
    32: 'End restrictions',     33: 'Turn right ahead',
    34: 'Turn left ahead',      35: 'Go straight',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right',           39: 'Keep left',
    40: 'Roundabout',           41: 'End no passing',
    42: 'End no passing >3.5t'
}

# YOUR ACTUAL KERAS MAPPING (alphabetical sort from training)
keras_class_mapping = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7,
    '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14,
    '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21,
    '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, '34': 28,
    '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35,
    '41': 36, '42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42
}
keras_index_to_classid = {v: int(k) for k, v in keras_class_mapping.items()}

# ========== CHANGE THIS PATH ==========
img_path = r"C:\Users\mohmm\Downloads\donwload (1).jpg"
# =====================================

img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Could not load image from {img_path}")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype('float32')

    pred = model.predict(img_array, verbose=0)
    keras_idx = np.argmax(pred)
    confidence = np.max(pred)

    actual_classid = keras_index_to_classid[keras_idx]
    label = class_names[actual_classid]

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title(f"{label}\nConfidence: {confidence:.1%}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Predicted: {label} (ClassId {actual_classid}, keras_idx {keras_idx})")
    print(f"Confidence: {confidence:.1%}")