import os
import tensorflow as tf

from dataset import build_dataset
from model import build_model
from utils import (
    total_loss,
    psnr_metric,
    ssim_metric
)

# =========================
# GLOBAL PATHS (SAME AS NOTEBOOK)
# =========================
INPUT_DIR  = "/content/dataset/inputs"
TARGET_DIR = "/content/dataset/targets"

# =========================
# TRAINING CONFIG
# =========================
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4

# =========================
# LOAD FILENAMES
# =========================
def list_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

inputs = list_images(INPUT_DIR)
targets = list_images(TARGET_DIR)

assert len(inputs) == len(targets), "Input/Target count mismatch"

# =========================
# TRAIN / VAL SPLIT
# =========================
train_files = inputs[:5000]
val_files   = inputs[5000:]

print("Train samples:", len(train_files))
print("Val samples  :", len(val_files))

# =========================
# BUILD DATASETS
# =========================
train_ds = build_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = build_dataset(
    val_files,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# BUILD MODEL
# =========================
model = build_model()

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=total_loss,
    metrics=[psnr_metric, ssim_metric]
)

# =========================
# CALLBACKS
# =========================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model_task1.keras",
    monitor="val_psnr_metric",
    mode="max",
    save_best_only=True,
    verbose=1
)

lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_psnr_metric",
    mode="max",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_psnr_metric",
    mode="max",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint_cb, lr_cb, earlystop_cb]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)
