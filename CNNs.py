import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Definisikan direktori data
base_dir = 'D:\data-train'  # Ganti dengan path folder Anda

# Parameter
img_width, img_height = 150, 150
batch_size = 32
epochs = 20
validation_split = 0.2  # 20% data untuk validasi

# Data Augmentation dan Pembagian Data
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split  # Menentukan split data validasi
)

# Generator untuk data training
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Data training
)

# Generator untuk data validation
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Data validation
)

# Membangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluasi model
eval_result = model.evaluate(validation_generator)
print(f"\nEvaluation Result: Loss = {eval_result[0]}, Accuracy = {eval_result[1]}")

# Simpan model
model_save_path = 'path/to/save/your/model/my_cnn_model.h5'  # Ganti dengan path untuk menyimpan model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
