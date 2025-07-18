import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ---------------- Paths ----------------
train_dir = r'C:\Users\DELL\OneDrive\Desktop\EmmotiSense\archive\train'
val_dir   = r'C:\Users\DELL\OneDrive\Desktop\EmmotiSense\archive\test'

# ---------------- Parameters ----------------
img_size = 48
batch_size = 64
num_classes = 7

# ---------------- Load Datasets ----------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

# Normalize pixel values
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)

# ---------------- Build CNN Model ----------------
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# ---------------- Compile ----------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- Train ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# ---------------- Save ----------------
model.save("emotion_model.h5")

# ---------------- Evaluate ----------------
test_loss, test_acc = model.evaluate(val_ds)
print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")