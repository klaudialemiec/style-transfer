from PIL import Image
import math
import streamlit as st
import tensorflow as tf
from style_content import StyleContentModel
from style_transfer import (
    load_img,
    image_to_tenforlow_image,
    train_step,
    tensor_to_image,
)


STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

CONTENT_LAYERS = ["block5_conv2"]


def upload_image(caption):
    image_buffer = st.file_uploader(caption)
    if image_buffer:
        image = Image.open(image_buffer)
        st.image(image, caption=caption)
        return image
    return None


def transfer_style(
    extractor,
    content_image,
    style_image,
    optimizer,
    num_epochs=5,
    steps_per_epochs=20,
):
    targets = {
        "style": extractor(style_image)["style"],
        "content": extractor(content_image)["content"],
    }
    len_layers = {"style": len(STYLE_LAYERS), "content": len(CONTENT_LAYERS)}
    image = tf.Variable(content_image)
    num_columns = math.ceil(num_epochs / 2)
    columns = st.beta_columns(num_columns)

    progress_bar = st.empty()

    for epoch in range(num_epochs):
        column_idx = epoch % num_columns

        for _ in range(steps_per_epochs):
            train_step(extractor, image, optimizer, targets, len_layers)

        progress_bar.write(f"Epoch {epoch+1}/{num_epochs} is done.")

        with columns[column_idx]:
            st.image(tensor_to_image(image), caption=f"Step {epoch+1}")

    progress_bar = ""

    st.image(tensor_to_image(image), caption="Final result", use_column_width=True)
    st.balloons()


st.title("Style transform")
column1, column2 = st.beta_columns(2)

with column1:
    content_img = upload_image("Content image")
    if content_img:
        content_image = image_to_tenforlow_image(content_img)

with column2:
    style_img = upload_image("Style image")
    if style_img:
        style_image = image_to_tenforlow_image(style_img)

start_button = st.button("Transfer it!")
if start_button:
    if content_img and style_img:
        st.text("Style trasfer has just started. It may take a few minutes.")
        extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)
        content_image = image_to_tenforlow_image(content_img)
        style_image = image_to_tenforlow_image(style_img)
        optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        transfer_style(extractor, content_image, style_image, optimizer)

    elif content_img is None and style_img is None:
        st.error("Content and style images were not given.")
    elif content_img is None:
        st.error("Content image was not given.")
    else:
        st.error("Style image was not given.")
