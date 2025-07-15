import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Configuración inicial
plt.style.use('ggplot')
sns.set()

IMG_SIZE = 150
MODEL_PATH = 'modelo_pneumonia.h5'

# Función para cargar imágenes
def load_data(data_dir):
    X = []
    Y = []
    for label in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(data_dir, label)
        for img_file in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Y.append(0 if label == "NORMAL" else 1)
    return np.array(X), np.array(Y)

# Cargar datasets
print("[INFO] Cargando imágenes...")
x_train, y_train = load_data('./chest_xray/train')
x_val, y_val = load_data('./chest_xray/val')
x_test, y_test = load_data('./chest_xray/test')

print("Datos cargados:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# Normalizar imágenes
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Redimensionar para TensorFlow (agregar canal de color)
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Verificar si el modelo ya existe
if os.path.exists(MODEL_PATH):
    print("[INFO] Cargando modelo guardado...")
    model = load_model(MODEL_PATH)
else:
    print("[INFO] Entrenando modelo desde cero...")
    # Construcción del modelo
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Salida binaria
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Reduce learning rate cuando se estanque
    lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=1e-5)

    # Entrenamiento
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=10,
                        validation_data=(x_val, y_val),
                        callbacks=[lr_reduction])

    # Guardar el modelo
    model.save(MODEL_PATH)
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")

    # Graficar evolución del entrenamiento
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# Evaluar el modelo
print("[INFO] Evaluando modelo...")
predictions = (model.predict(x_test) > 0.5).astype("int32")

print(classification_report(y_test, predictions))

# Matriz de confusión
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Función para probar con una sola imagen
def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(img)
    if pred >= 0.5:
        print(f"[RESULT] Probabilidad de Neumonía: {pred[0][0]:.4f} --> PNEUMONIA DETECTADA")
    else:
        print(f"[RESULT] Probabilidad de Neumonía: {pred[0][0]:.4f} --> PULMÓN SANO")

# Ejemplo con una imagen del test set
predict_single_image('./chest_xray/test/PNEUMONIA/person8_virus_28.jpeg')
