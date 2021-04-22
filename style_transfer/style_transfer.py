# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:40:18 2021
@author: sengupta
"""
import tensorflow as tf
import matplotlib.pyplot as plt

def vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  network = tf.keras.Model(inputs=[vgg.input], outputs = outputs )
  return network

def gram_matrix(layer_activation):
  result = tf.linalg.einsum('bijc,bijd->bcd', layer_activation, layer_activation)
  input_shape = tf.shape(layer_activation)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

  return result / num_locations


class StyleContentModels(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super().__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs * 255
    preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_inputs)
    style_outputs = outputs[:self.num_style_layers]
    content_outputs = outputs[self.num_style_layers:]

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
    style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}

print("point 1")

content_layers = ["block4_conv2"]
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

content_image = tf.keras.preprocessing.image.load_img('C:\\Users\\sengupta\\OneDrive - Qualcomm\\Documents\\mlCode\\dataset\\content.jpg')
plt.imshow(content_image)
content_image= tf.keras.preprocessing.image.img_to_array(content_image)
type(content_image),content_image.shape,content_image.min(), content_image.max()
content_image=content_image/content_image.max()

print("point 2")

style_image=tf.keras.preprocessing.image.load_img('C:\\Users\\sengupta\\OneDrive - Qualcomm\\Documents\\mlCode\\dataset\\style.jpg')
plt.imshow(style_image)
style_image=tf.keras.preprocessing.image.img_to_array(style_image)
type(style_image),style_image.shape,style_image.min(), style_image.max()
style_image=style_image/style_image.max()

print("point 3")

content_image=content_image[tf.newaxis,:]
style_image=style_image[tf.newaxis,:]

num_style_layers = len(style_layers)
num_content_layers = len(content_layers)

print("point 4")
extractor = StyleContentModels(style_layers=style_layers, content_layers=content_layers)

print("point 5")
results = extractor(content_image)

content_target = extractor(content_image)['content']
style_target = extractor(style_image)['style']

new_image = tf.Variable(content_image)

content_weight = 1
style_weight = 100
optimizer = tf.optimizers.Adam(learning_rate=0.02)

epochs = 100
print_every = 1

for epoch in range(epochs):
  with tf.GradientTape() as tape:
    outputs = extractor(new_image)

    content_outputs = outputs['content']
    style_outputs = outputs['style']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) **2 ) for name in style_outputs.keys()])
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2) for name in content_outputs.keys()])

    total_loss = content_loss * content_weight / num_content_layers + style_loss * style_weight / num_style_layers
  gradient = tape.gradient(total_loss, new_image)
  optimizer.apply_gradients([(gradient, new_image)])

  new_image.assign(tf.clip_by_value(new_image, 0.0, 1.0))

  if (epochs + 1) % print_every == 0:
    print("epoch : {}, content loss : {}, style loss : {}, total loss : {}".format(epoch+1, content_loss, style_loss, total_loss))

  if (epochs + 1) % 25 == 0 :
    plt.imshow(tf.squeeze(new_image, axis=0))
    plt.show()


plt.imshow(tf.squeeze(new_image, axis=0))
plt.show()
