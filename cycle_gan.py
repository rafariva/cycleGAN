#https://rubikscode.net/2019/02/11/implementing-cyclegan-using-python/

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras modules
from keras.layers import Input, LeakyReLU, UpSampling2D, Conv2D, Concatenate
from keras_contrib.layers.normalization import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam

class CycleGAN():

    ## cycle_gan_constructor.py
    def __init__(self, image_shape, cycle_lambda, image_hepler):
        self.optimizer = Adam(0.0002, 0.5)
        
        self.cycle_lambda = cycle_lambda 
        self.id_lambda = 0.1 * self.cycle_lambda
        self._image_helper = image_hepler
        self.img_shape = image_shape
        
        # Calculate output shape
        patch = int(self.img_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        print("Build Discriminators...")
        self._discriminatorX = self._build_discriminator_model()
        self._compile_discriminator_model(self._discriminatorX)
        self._discriminatorY = self._build_discriminator_model()
        self._compile_discriminator_model(self._discriminatorY)
        
        print("Build Generators...")
        self._generatorXY = self._build_generator_model()
        self._generatorYX = self._build_generator_model()        
        
        print("Build GAN...")
        self._build_and_compile_gan()
    #######constructor##########

    ##  cycle_gan_train.py
    def train(self, epochs, batch_size, train_data_path):
        
        real = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        history = []
        
        for epoch in range(epochs):
            for i, (imagesX, imagesY) in enumerate(self._image_helper.load_batch_of_train_images(train_data_path, batch_size)):
                print ("---------------------------------------------------------")
                print ("******************Epoch {} | Batch {}***************************".format(epoch, i))
                print("Generate images...")
                fakeY = self._generatorXY.predict(imagesX)
                fakeX = self._generatorYX.predict(imagesY)

                print("Train Discriminators...")
                discriminatorX_loss_real = self._discriminatorX.train_on_batch(imagesX, real)
                discriminatorX_loss_fake = self._discriminatorX.train_on_batch(fakeX, fake)
                discriminatorX_loss = 0.5 * np.add(discriminatorX_loss_real, discriminatorX_loss_fake)

                discriminatorY_loss_real = self._discriminatorY.train_on_batch(imagesY, real)
                discriminatorY_loss_fake = self._discriminatorY.train_on_batch(fakeY, fake)
                discriminatorY_loss = 0.5 * np.add(discriminatorY_loss_real, discriminatorY_loss_fake)

                mean_discriminator_loss = 0.5 * np.add(discriminatorX_loss, discriminatorY_loss)
                
                print("Train Generators...")
                generator_loss = self.gan.train_on_batch([imagesX, imagesY],
                                                        [real, real,
                                                        imagesX, imagesY,
                                                        imagesX, imagesY])

                print ("Discriminator loss: {}".format(mean_discriminator_loss[0]))
                print ("Generator loss: {}".format(generator_loss[0]))
                print ("---------------------------------------------------------")
                
                history.append({"D":mean_discriminator_loss[0],"G":generator_loss})
                
                if i%100 ==0:
                    self._save_images("{}_{}".format(epoch, i), train_data_path)

        self._plot_loss(history)
    ######train######


    ## cycle_gan_build_generator_model.py
    def _encode__layer(self, input_layer, filters):
        layer = Conv2D(filters, kernel_size=4, strides=2, padding='same')(input_layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = InstanceNormalization()(layer)
        return layer
        
    def _decode_transform_layer(self, input_layer, forward_layer, filters):
        layer = UpSampling2D(size=2)(input_layer)
        layer = Conv2D(filters, kernel_size=4, strides=1, padding='same', activation='relu')(layer)
        layer = InstanceNormalization()(layer)
        layer = Concatenate()([layer, forward_layer])
        return layer
    
    def _build_generator_model(self):
        generator_input = Input(shape=self.img_shape)
        
        print("Build Encoder...")
        encode_layer_1 = self._encode__layer(generator_input, 32);
        encode_layer_2 = self._encode__layer(encode_layer_1, 64);
        encode_layer_3 = self._encode__layer(encode_layer_2, 128);
        encode_layer_4 = self._encode__layer(encode_layer_3, 256);
        
        print("Build Transformer - Decoder...")
        decode_transform_layer1 = self._decode_transform_layer(encode_layer_4, encode_layer_3, 128);
        decode_transform_layer2 = self._decode_transform_layer(decode_transform_layer1, encode_layer_2, 64);
        decode_transform_layer3 = self._decode_transform_layer(decode_transform_layer2, encode_layer_1, 32);
        
        generator_model = UpSampling2D(size = 2)(decode_transform_layer3)
        generator_model = Conv2D(self.img_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(generator_model)
        
        final_generator_model = Model(generator_input, generator_model)
        final_generator_model.summary()
        return final_generator_model
    ########generator##########

        
    ## cycle_gan_build_discriminator_model.py
    def _build_discriminator_model(self):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_model = Conv2D(64, kernel_size=4, strides=2, padding='same')(discriminator_input)
        discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)
        discriminator_model = Conv2D(128, kernel_size=4, strides=2, padding='same')(discriminator_model)
        discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)
        discriminator_model = InstanceNormalization()(discriminator_model)
        discriminator_model = Conv2D(256, kernel_size=4, strides=2, padding='same')(discriminator_model)
        discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)
        discriminator_model = InstanceNormalization()(discriminator_model)
        discriminator_model = Conv2D(512, kernel_size=4, strides=2, padding='same')(discriminator_model)
        discriminator_model = LeakyReLU(alpha=0.2)(discriminator_model)
        discriminator_model = InstanceNormalization()(discriminator_model)        
        discriminator_model = Conv2D(1, kernel_size=4, strides=1, padding='same')(discriminator_model)
        
        return Model(discriminator_input, discriminator_model)
    #########discriminator############

    ## cycle_gan_build_and_compile_gan.py
    def _compile_discriminator_model(self, model):
        model.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        model.summary()
    
    def _build_and_compile_gan(self):
        
        imageX = Input(shape=self.img_shape)
        imageY = Input(shape=self.img_shape)

        fakeY = self._generatorXY(imageX)
        fakeX = self._generatorYX(imageY)

        reconstructedX = self._generatorYX(fakeY)
        reconstructedY = self._generatorXY(fakeX)
        
        imageX_id = self._generatorYX(imageX)
        imageY_id = self._generatorXY(imageY)
        
        self._discriminatorX.trainable = False
        self._discriminatorY.trainable = False
        
        validX = self._discriminatorX(fakeX)
        validY = self._discriminatorY(fakeY)
        
        self.gan = Model(inputs=[imageX, imageY],
                          outputs=[ validX, validY,
                                    reconstructedX, reconstructedY,
                                    imageX_id, imageY_id ])
        self.gan.compile(loss=['mse', 'mse',
                                'mae', 'mae',
                                'mae', 'mae'],
                        loss_weights=[  1, 1,
                                        self.cycle_lambda, self.cycle_lambda,
                                        self.id_lambda, self.id_lambda ],
                        optimizer=self.optimizer)
        
        self.gan.summary()
    #####build and compile##########
    
    def _save_images(self, epoch, path):
        (img_X, img_Y) = self._image_helper.load_testing_image(path)
        
        fake_Y = self._generatorXY.predict(img_X)
        fake_X = self._generatorYX.predict(img_Y)

        plot_images = np.concatenate([img_X, fake_Y, img_Y, fake_X])

        # Rescale
        plot_images = 0.5 * plot_images + 0.5
        self._image_helper.save_image(plot_images, epoch)

        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()