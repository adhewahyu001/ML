import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Definisi jalur direktori yang berisi gambar
sdir = r'D:\data_labeling'

# Daftar kosong untuk menyimpan jalur file gambar dan labelnya
filepaths = []
labels = []

# Mendapatkan daftar nama sub-direktori (kelas) dalam direktori sdir
classlist = os.listdir(sdir)

# Iterasi melalui setiap sub-direktori (kelas)
for klass in classlist:
    classpath = os.path.join(sdir, klass)  # Jalur lengkap ke sub-direktori
    if os.path.isdir(classpath):  # Memeriksa apakah ini adalah direktori
        flist = os.listdir(classpath)  # Mendapatkan daftar file dalam sub-direktori
        for f in flist:
            fpath = os.path.join(classpath, f)  # Jalur lengkap ke file gambar
            filepaths.append(fpath)  # Menambahkan jalur file ke daftar filepaths
            labels.append(klass)  # Menambahkan label kelas ke daftar labels

# Membuat Pandas Series dari daftar filepaths dan labels
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')

# Menggabungkan kedua Series menjadi dataframe df
df = pd.concat([Fseries, Lseries], axis=1)

# Mencetak lima baris pertama dari dataframe df
print(df.head())

# Menghitung dan mencetak jumlah gambar dalam setiap kelas (label)
print(df['labels'].value_counts())

# Proporsi data pelatihan, pengujian, dan validasi
train_split = 0.8
test_split = 0.1

# Menghitung proporsi data validasi
dummy_split = test_split / (1 - train_split)

# Membagi data menjadi data pelatihan dan "dummy" data pengujian
train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)

# Membagi "dummy" data pengujian menjadi data pengujian dan data validasi
test_df, valid_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)

# Mencetak panjang data dari setiap bagian
print('train_df length:', len(train_df), 'test_df length:', len(test_df), 'valid_df length:', len(valid_df))

# Persiapkan generator data untuk data pelatihan, validasi, dan pengujian
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                                    target_size=(224, 224), class_mode='binary', batch_size=8)

valid_generator = valid_datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                                    target_size=(224, 224), class_mode='binary', batch_size=8)

test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                                  target_size=(224, 224), class_mode='binary', batch_size=8)

# Membangun model ResNet-152
base_model = ResNet152(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model_resnet152 = Model(inputs=base_model.input, outputs=predictions)

# Membekukan layer-layer basis ResNet-152
for layer in base_model.layers:
    layer.trainable = False

model_resnet152.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pelatihan model
checkpoint = ModelCheckpoint('best_model_resnet152.keras', monitor='val_accuracy', save_best_only=True, mode='max')

history = model_resnet152.fit(train_generator,
                              epochs=5,
                              validation_data=valid_generator,
                              callbacks=[checkpoint])

# Mencetak hasil pelatihan
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

# Evaluasi pengujian
best_model_resnet152 = tf.keras.models.load_model('best_model_resnet152.keras')
test_loss, test_acc = best_model_resnet152.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Membuat prediksi dengan model terbaik
y_pred = best_model_resnet152.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = test_generator.classes

# Mencetak laporan klasifikasi
print('Classification Report')
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Membuat confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model_resnet152.save('model_resnet152.h5')
