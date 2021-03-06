import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses

def res_seq(num_filters):
  return tf.keras.Sequential([
    layers.Conv2D(num_filters, (1, 1)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(num_filters, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(num_filters, (1, 1)),
    layers.BatchNormalization()
  ])

def ResidualBlock(input, num_filters):
  # x = res_seq(num_filters)(input)
  x = layers.Conv2D(num_filters, (1, 1))(input)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(num_filters, (1, 1))(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input])
  return layers.Activation('relu')(x)

# class ResidualBlock(tf.keras.Model):
#   def __init__(self, filters):
#     super(ResidualBlock, self).__init__(name='')
#     self.conv1 = layers.Conv2D(filters, (3, 3), padding='same')
#     self.bn1 = layers.BatchNormalization()
#     self.act1 = layers.Activation('relu')
#     self.conv2 = layers.Conv2D(filters, (3, 3), padding='same')
#     self.bn2 = layers.BatchNormalization()
#     self.act2 = layers.Activation('relu')
#     self.conv3 = layers.Conv2D(filters, (3, 3), padding='same')
#     self.bn3 = layers.BatchNormalization()
  
#   def call(self, input, training=False):
#     x = self.conv1(input)
#     x = self.bn1(x, training=training)
#     x = self.act1(x)
#     x = self.conv2(input)
#     x = self.bn2(x, training=training)
#     x = self.act2(x)
#     x = self.conv3(x)
#     x = self.bn3(x, training=training)
#     x = layers.Add()([x, input])
#     x = layers.Activation('relu')(x)
#     return x

def conv_block(input_tensor, num_filters, do=0):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Dropout(do)(encoder) if do else encoder
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Dropout(do)(encoder) if do else encoder
  return encoder

def encoder_block(input_tensor, num_filters, do ,res=False):
  if res:
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = ResidualBlock(encoder, num_filters)
  else:
    encoder = conv_block(input_tensor, num_filters, do)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters, padding, do):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=padding)(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Dropout(do)(decoder) if do else decoder
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Dropout(do)(decoder) if do else decoder
  return decoder

def Model(res=False):
  inputs = layers.Input(shape=(128,128,1))
  # encode network
  # 64*64*16 , 101
  encoder_pool1, encoder1 = encoder_block(inputs, 16, 0.1, res)
  # 32*32*32, 50
  encoder_pool2, encoder2 = encoder_block(encoder_pool1, 32, 0.1, res)
  # 16*16*64, 25
  encoder_pool3, encoder3 = encoder_block(encoder_pool2, 64, 0.1, res)
  # 8*8*128, 12
  encoder_pool4, encoder4 = encoder_block(encoder_pool3, 128, 0.1, res)
  # 4 * 4 * 256, 
  encoder_pool5, encoder5 = encoder_block(encoder_pool4, 256, 0.1, res)
  # center, 4*4*512
  center = conv_block(encoder_pool5, 512, 0.1)
  # decode network
  # 8*8*256
  dec0 = decoder_block(center, encoder5, 256, 'same', 0.1)
  # 16*16*128
  dec1 = decoder_block(dec0, encoder4, 128, 'same', 0.1)
  # 32*32*64
  dec2 = decoder_block(dec1, encoder3, 64, 'same', 0.1)
  # 64*64*32
  dec3 = decoder_block(dec2, encoder2, 32, 'same', 0.1)
  # 128*128*16
  dec4 = decoder_block(dec3, encoder1, 16,'same', 0.1)
  # 128*128*1
  outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(dec4)
  return models.Model(inputs=[inputs], outputs=[outputs])

if __name__ == '__main__':
  m = Model(res=True)
  print(m.summary())