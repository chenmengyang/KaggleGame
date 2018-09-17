from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses

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

def encoder_block(input_tensor, num_filters, do):
  encoder = conv_block(input_tensor, num_filters, do)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters, padding, do):
  decoder = layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding=padding)(input_tensor)
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

def Model():
    inputs = layers.Input(shape=(128,128,1))
    # encode network
    # 64*64*16 , 101
    encoder_pool1, encoder1 = encoder_block(inputs, 16, 0.1)
    # 32*32*32, 50
    encoder_pool2, encoder2 = encoder_block(encoder_pool1, 32, 0.1)
    # 16*16*64, 25
    encoder_pool3, encoder3 = encoder_block(encoder_pool2, 64, 0.1)
    # 8*8*128, 12
    encoder_pool4, encoder4 = encoder_block(encoder_pool3, 128, 0.1)
    # 4 * 4 * 256, 
    encoder_pool5, encoder5 = encoder_block(encoder_pool4, 256, 0.1)
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