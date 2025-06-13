## MediaPipe - Configuraciones Avanzadas

### **Teoría**: MediaPipe permite ajustar sensibilidad y rendimiento según el contexto

```python
# Para ambientes con poca luz o movimiento rápido
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.3,  # Más sensible
    min_tracking_confidence=0.3,   # Menos estricto en seguimiento
    model_complexity=0             # Más rápido, menos preciso
)

# Para mayor precisión sacrificando velocidad
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,         # 468 landmarks vs 68
    min_detection_confidence=0.8,  # Más estricto
    model_complexity=1             # Más lento, más preciso
)
```

## Landmarks Críticos por Funcionalidad

### **Teoría**: Los landmarks están numerados consistentemente, cada uno representa un punto anatómico específico

```python
# Detección de parpadeo - EAR (Eye Aspect Ratio)
EYE_LANDMARKS = {
    'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
}

# Puntos específicos para EAR
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # Para cálculo de apertura
LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
```

### **Teoría**: Los gestos se detectan por relaciones espaciales entre landmarks

```python
# Detección de "OK" con la mano
def detect_ok_gesture(landmarks):
    # Pulgar (4) y índice (8) cercanos, otros dedos extendidos
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    
    # Otros dedos extendidos
    extended_fingers = [landmarks[12][1] < landmarks[10][1],  # Medio
                       landmarks[16][1] < landmarks[14][1],   # Anular
                       landmarks[20][1] < landmarks[18][1]]   # Meñique
    
    return distance < 30 and all(extended_fingers)
```

## Cálculos Matemáticos Específicos

### **Teoría**: EAR (Eye Aspect Ratio) es más robusto que distancia simple para parpadeo

```python
def calculate_ear(eye_landmarks):
    # Distancias verticales
    A = math.hypot(eye_landmarks[1][0] - eye_landmarks[5][0], 
                   eye_landmarks[1][1] - eye_landmarks[5][1])
    B = math.hypot(eye_landmarks[2][0] - eye_landmarks[4][0], 
                   eye_landmarks[2][1] - eye_landmarks[4][1])
    # Distancia horizontal
    C = math.hypot(eye_landmarks[0][0] - eye_landmarks[3][0], 
                   eye_landmarks[0][1] - eye_landmarks[3][1])
    
    # EAR = (A + B) / (2.0 * C)
    ear = (A + B) / (2.0 * C)
    return ear

# EAR < 0.25 indica ojo cerrado típicamente
```

### **Teoría**: MAR (Mouth Aspect Ratio) para detectar bostezos o habla

```python
def calculate_mar(mouth_landmarks):
    # Puntos verticales de la boca
    A = math.hypot(mouth_landmarks[2][0] - mouth_landmarks[10][0],
                   mouth_landmarks[2][1] - mouth_landmarks[10][1])
    B = math.hypot(mouth_landmarks[4][0] - mouth_landmarks[8][0],
                   mouth_landmarks[4][1] - mouth_landmarks[8][1])
    C = math.hypot(mouth_landmarks[6][0] - mouth_landmarks[6][0],
                   mouth_landmarks[6][1] - mouth_landmarks[6][1])
    # Distancia horizontal
    D = math.hypot(mouth_landmarks[0][0] - mouth_landmarks[6][0],
                   mouth_landmarks[0][1] - mouth_landmarks[6][1])
    
    mar = (A + B + C) / (3.0 * D)
    return mar
```

## Preprocessamiento de Imágenes

### **Teoría**: La normalización afecta cómo el modelo interpreta los datos

```python
# Normalización para modelos entrenados con ImageNet
def preprocess_imagenet(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    # ImageNet mean y std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image/255.0 - mean) / std
    return np.expand_dims(image, axis=0)

# Para modelos custom con rango [-1, 1]
def preprocess_custom(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1  # Rango [-1, 1]
    return np.expand_dims(image, axis=0)
```

### **Teoría**: Data augmentation mejora generalización del modelo

```python
def augment_image(image):
    # Rotación aleatoria
    angle = np.random.uniform(-15, 15)
    center = (image.shape[1]//2, image.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    # Cambio de brillo
    brightness = np.random.uniform(0.8, 1.2)
    bright = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
    
    # Ruido gaussiano
    noise = np.random.normal(0, 10, bright.shape).astype(np.uint8)
    noisy = cv2.add(bright, noise)
    
    return noisy
```

## Sistemas de Estados Avanzados

### **Teoría**: Máquinas de estado finitas previenen transiciones erráticas

```python
class GestureStateMachine:
    def __init__(self):
        self.states = ['IDLE', 'DETECTING', 'CONFIRMED', 'COOLDOWN']
        self.current_state = 'IDLE'
        self.detection_count = 0
        self.confirmation_threshold = 5
        self.cooldown_frames = 30
        self.frame_counter = 0
    
    def update(self, gesture_detected):
        if self.current_state == 'IDLE':
            if gesture_detected:
                self.current_state = 'DETECTING'
                self.detection_count = 1
        
        elif self.current_state == 'DETECTING':
            if gesture_detected:
                self.detection_count += 1
                if self.detection_count >= self.confirmation_threshold:
                    self.current_state = 'CONFIRMED'
                    return True  # Gesto confirmado
            else:
                self.current_state = 'IDLE'
                self.detection_count = 0
        
        elif self.current_state == 'CONFIRMED':
            self.current_state = 'COOLDOWN'
            self.frame_counter = 0
        
        elif self.current_state == 'COOLDOWN':
            self.frame_counter += 1
            if self.frame_counter >= self.cooldown_frames:
                self.current_state = 'IDLE'
        
        return False
```

## Filtros y Suavizado

### **Teoría**: Los filtros reducen ruido y falsos positivos en detecciones

```python
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []
    
    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

class ExponentialFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_value = None
    
    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
```

## Manejo de Confianza y Umbrales Adaptativos

### **Teoría**: Umbrales adaptativos mejoran robustez en diferentes condiciones

```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold=0.5, adaptation_rate=0.01):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.recent_confidences = []
        self.window_size = 50
    
    def update_threshold(self, confidence, detection_result):
        self.recent_confidences.append(confidence)
        if len(self.recent_confidences) > self.window_size:
            self.recent_confidences.pop(0)
        
        avg_confidence = np.mean(self.recent_confidences)
        
        # Ajustar umbral basado en rendimiento
        if detection_result and confidence < self.threshold:
            # Falso negativo, reducir umbral
            self.threshold -= self.adaptation_rate
        elif not detection_result and confidence > self.threshold:
            # Falso positivo, aumentar umbral
            self.threshold += self.adaptation_rate
        
        # Mantener en rango válido
        self.threshold = np.clip(self.threshold, 0.1, 0.9)
        return self.threshold
```

## Validación y Debugging

### **Teoría**: Logging estructurado facilita debugging y optimización

```python
class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.detection_counts = {'true_positive': 0, 'false_positive': 0, 
                               'true_negative': 0, 'false_negative': 0}
    
    def log_frame_time(self, start_time):
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def get_fps(self):
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
    
    def log_detection(self, predicted, actual):
        if predicted and actual:
            self.detection_counts['true_positive'] += 1
        elif predicted and not actual:
            self.detection_counts['false_positive'] += 1
        elif not predicted and actual:
            self.detection_counts['false_negative'] += 1
        else:
            self.detection_counts['true_negative'] += 1
```

## Calibración Automática

### **Teoría**: La calibración automática adapta el sistema al usuario específico

```python
class AutoCalibrator:
    def __init__(self, calibration_frames=100):
        self.calibration_frames = calibration_frames
        self.baseline_measurements = []
        self.is_calibrated = False
        self.frame_count = 0
    
    def add_measurement(self, measurement):
        if not self.is_calibrated:
            self.baseline_measurements.append(measurement)
            self.frame_count += 1
            
            if self.frame_count >= self.calibration_frames:
                self.baseline_mean = np.mean(self.baseline_measurements)
                self.baseline_std = np.std(self.baseline_measurements)
                self.is_calibrated = True
                print(f"Calibración completa: mean={self.baseline_mean:.3f}, std={self.baseline_std:.3f}")
    
    def is_anomaly(self, measurement, threshold_std=2.0):
        if not self.is_calibrated:
            return False
        
        z_score = abs(measurement - self.baseline_mean) / self.baseline_std
        return z_score > threshold_std
```