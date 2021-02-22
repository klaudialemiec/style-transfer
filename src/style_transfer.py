import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def image_to_tenforlow_image(image):
    max_dim = 512
    img = np.array(image)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(
    outputs,
    targets,
    num_layers,
    content_weight=1e4,
    style_weight=1e-2,
):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_target = targets["style"]
    content_target = targets["content"]
    style_num_layers = num_layers["style"]
    content_num_layers = num_layers["content"]

    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / style_num_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / content_num_layers
    loss = style_loss + content_loss
    return loss


def train_step(
    extractor, image, optimizer, targets, num_layers, total_variation_weight=30
):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, targets, num_layers)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)
