from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19
import tensorflow as tf
from style_transfer import gram_matrix


class StyleContentModel(Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.vgg = self.vgg_model()
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }
        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}

    def vgg_model(self):
        layers = self.style_layers + self.content_layers
        vgg = VGG19(include_top=False, weights="imagenet")
        outputs = [vgg.get_layer(layer).output for layer in layers]
        model = Model([vgg.input], outputs)
        return model
