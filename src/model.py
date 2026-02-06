import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# GLOBAL CONFIG (AS IN NOTEBOOK)
# =========================
IMG_SIZE = 256
IN_CHANNELS = 3
BASE_CHANNELS = 64

TRANSFORMER_DIM = 256
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2


# =========================
# ENCODER BLOCK
# =========================
def encoder_block(x, filters, downsample=True):
    if downsample:
        x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
    else:
        x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.ReLU()(x)
    return x


# =========================
# TRANSFORMER BOTTLENECK
# =========================
def transformer_bottleneck(x):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]

    x_flat = layers.Reshape((h * w, c))(x)
    x_proj = layers.Dense(TRANSFORMER_DIM)(x_flat)

    for _ in range(NUM_TRANSFORMER_LAYERS):
        attn = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=TRANSFORMER_DIM
        )(x_proj, x_proj)
        x_proj = layers.LayerNormalization()(x_proj + attn)

        mlp = layers.Dense(TRANSFORMER_DIM * 2, activation="relu")(x_proj)
        mlp = layers.Dense(TRANSFORMER_DIM)(mlp)
        x_proj = layers.LayerNormalization()(x_proj + mlp)

    x_back = layers.Dense(c)(x_proj)
    x_out = layers.Reshape((h, w, c))(x_back)
    return x_out


# =========================
# DECODER BLOCK
# =========================
def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    return x


# =========================
# BUILD MODEL
# =========================
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IN_CHANNELS))

    # Encoder
    e1 = encoder_block(inputs, BASE_CHANNELS, downsample=False)
    e2 = encoder_block(e1, BASE_CHANNELS * 2)
    e3 = encoder_block(e2, BASE_CHANNELS * 4)
    e4 = encoder_block(e3, BASE_CHANNELS * 8)

    # Transformer bottleneck
    t = transformer_bottleneck(e4)

    # Decoder
    d3 = decoder_block(t, e3, BASE_CHANNELS * 4)
    d2 = decoder_block(d3, e2, BASE_CHANNELS * 2)
    d1 = decoder_block(d2, e1, BASE_CHANNELS)

    outputs = layers.Conv2D(3, 1, activation="sigmoid")(d1)

    return models.Model(inputs, outputs)
