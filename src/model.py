def encoder_block(x, filters, downsample=False):
    if downsample:
        x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
    else:
        x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    return x
def build_encoder(inputs):
    # E1
    e1 = encoder_block(inputs, BASE_CHANNELS, downsample=False)   # (H, W, 64)

    # E2
    e2 = encoder_block(e1, BASE_CHANNELS*2, downsample=True)      # (H/2, W/2, 128)

    # E3
    e3 = encoder_block(e2, BASE_CHANNELS*4, downsample=True)      # (H/4, W/4, 256)

    # E4 (Transformer input)
    e4 = encoder_block(e3, BASE_CHANNELS*8, downsample=True)      # (H/8, W/8, 512)

    return e1, e2, e3, e4

#-------------------------------------------------------------------------------
def transformer_block(x):
    # Self-attention
    attn = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=TRANSFORMER_DIM
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # MLP
    mlp = layers.Dense(TRANSFORMER_DIM * 2, activation="relu")(x)
    mlp = layers.Dense(TRANSFORMER_DIM)(mlp)

    x = layers.Add()([x, mlp])
    x = layers.LayerNormalization()(x)

    return x
def transformer_bottleneck(x):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]

    # Flatten to tokens
    x = layers.Reshape((h * w, c))(x)

    # Project down
    x = layers.Dense(TRANSFORMER_DIM)(x)

    # Transformer layers
    for _ in range(NUM_TRANSFORMER_LAYERS):
        x = transformer_block(x)

    # Project back
    x = layers.Dense(c)(x)

    # Reshape back to feature map
    x = layers.Reshape((h, w, c))(x)

    return x

    #-------------------------------------------------
def decoder_block(x, skip, filters):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    return x
#-----------------------------------------------------------------------
def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IN_CHANNELS))

    # Encoder
    e1, e2, e3, e4 = build_encoder(inputs)

    # Transformer bottleneck
    t = transformer_bottleneck(e4)

    # Decoder
    d3 = decoder_block(t, e3, BASE_CHANNELS*4)
    d2 = decoder_block(d3, e2, BASE_CHANNELS*2)
    d1 = decoder_block(d2, e1, BASE_CHANNELS)

    # Output
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(d1)

    return models.Model(inputs, outputs)
    
model = build_model()
model.summary()


