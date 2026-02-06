import tensorflow as tf
from tensorflow.keras.applications import VGG16

# =========================
# BASIC LOSSES
# =========================
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)


# =========================
# PERCEPTUAL LOSS (VGG)
# =========================
vgg = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3)
)
vgg.trainable = False

vgg_layer_names = ["block3_conv3"]
vgg_outputs = [vgg.get_layer(name).output for name in vgg_layer_names]
vgg_model = tf.keras.Model(vgg.input, vgg_outputs)


def perceptual_loss(y_true, y_pred):
    y_true_vgg = vgg_model(y_true)
    y_pred_vgg = vgg_model(y_pred)
    return tf.reduce_mean(tf.abs(y_true_vgg - y_pred_vgg))


# =========================
# COLOR CONSISTENCY LOSS
# =========================
def color_loss(y_true, y_pred):
    mean_true, var_true = tf.nn.moments(y_true, axes=[1, 2])
    mean_pred, var_pred = tf.nn.moments(y_pred, axes=[1, 2])

    std_true = tf.sqrt(var_true + 1e-6)
    std_pred = tf.sqrt(var_pred + 1e-6)

    mean_loss = tf.reduce_mean(tf.abs(mean_true - mean_pred))
    std_loss  = tf.reduce_mean(tf.abs(std_true - std_pred))

    return mean_loss + std_loss


# =========================
# TOTAL LOSS (TRAINING)
# =========================
def total_loss(y_true, y_pred):
    l1   = l1_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    perc = perceptual_loss(y_true, y_pred)
    col  = color_loss(y_true, y_pred)

    return (
        0.8  * l1 +
        0.1  * ssim +
        0.05 * perc +
        0.05 * col
    )


# =========================
# METRICS
# =========================
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.psnr(y_true, y_pred, max_val=1.0)
    )


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )
