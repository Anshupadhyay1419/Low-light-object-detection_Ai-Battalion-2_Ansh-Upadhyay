import tensorflow as tf
import os

INPUT_DIR  = "/content/dataset/inputs"
TARGET_DIR = "/content/dataset/targets"

IMG_SIZE = 256


def load_image_pair(inp_path, tgt_path):
    inp = tf.io.read_file(inp_path)
    inp = tf.image.decode_jpeg(inp, channels=3)
    inp = tf.image.resize(inp, (IMG_SIZE, IMG_SIZE))
    inp = tf.cast(inp, tf.float32) / 255.0

    tgt = tf.io.read_file(tgt_path)
    tgt = tf.image.decode_jpeg(tgt, channels=3)
    tgt = tf.image.resize(tgt, (IMG_SIZE, IMG_SIZE))
    tgt = tf.cast(tgt, tf.float32) / 255.0

    return inp, tgt


def build_dataset(file_list, batch_size=4, shuffle=False):
    inp_paths = [os.path.join(INPUT_DIR, f) for f in file_list]
    tgt_paths = [os.path.join(TARGET_DIR, f) for f in file_list]

    ds = tf.data.Dataset.from_tensor_slices((inp_paths, tgt_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda i, t: load_image_pair(i, t),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
