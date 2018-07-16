#Mini Sprites - Autoencoder

#Imports
import keras
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import h5py
import time
from sklearn.cluster import KMeans
from math import floor

#Import Images - Now in HSV
# [Num, Rows, Cols, Channels]
def fix(imput):
    output = np.zeros([100, 100, 3])
    for i in range(100):
        for j in range(100):
            output[i][j] = [imput[i][j][0],
                            imput[i][j][0],
                            imput[i][j][0]]
    
    return output

def fix2(imput):
    output = np.zeros([20, 20, 3])
    for i in range(20):
        for j in range(20):
            output[i][j] = [imput[i][j][0],
                            imput[i][j][0],
                            imput[i][j][0]]
    
    return output

Images_BW = []
Images_CL = []

def noise(n = 1):
    return np.random.uniform(-1.0, 1.0, size = [n, 3, 3, 128])

def noise2():
    output = np.random.uniform(0.0, 1.0, size = [128])
    
    output = np.tile(output, [5, 5, 1])
    
    return output

def one():
    return np.random.uniform(0.95, 1.0, size = [1])

def zero():
    return np.random.uniform(0.0, 0.05, size = [1])

def top_colors(img):
    
    pixels = []
    
    for i in range(100):
        for j in range(100):
            if(img[i][j][0] >= 0.001 or img[i][j][1] >= 0.001 or img[i][j][2] >= 0.001):
                pixels.append(img[i][j][...,:3])


    kmeans = KMeans(n_clusters=3).fit(pixels)
    
    return kmeans.cluster_centers_.flatten()

def hueShift(arr, amount):
    hsv = rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = hsv_to_rgb(hsv)
    return rgb

def process(img):
    return hueShift(img, random.random())

print("Importing Images...")


for n in range(0, 0): #Otherwise 721
    temp1 = Image.open("Sprites/sp ("+str(n)+").png")
                
    temp = np.array(temp1, dtype='float32')
    tmp = np.array(temp1.convert('HSV'), dtype='float32')
    outti = np.zeros([100, 100, 1])

    #Check that 0 alpha are gone
    for i in range(100):
        for j in range(100):
            if(temp[i][j][3] <= 25):
                outti[i][j][0] = 0
                temp[i][j] = np.zeros((4))
            else:
                outti[i][j][0] = max(0.75 - (tmp[i][j][2] / 255), 0) * 1.333
                temp[i][j][0] = temp[i][j][0]/255.0
                temp[i][j][1] = temp[i][j][1]/255.0
                temp[i][j][2] = temp[i][j][2]/255.0
    
    #plt.imshow(color.hsv_to_rgb(tmp))
    #plt.show()
    
    Images_BW.append(outti)
    Images_CL.append(temp[...,:3])
    
    if n % 72 == 0 and n != 0:
        print(str(round(n/72)) + "0%")
        

#Fix to np array
Images_BW = np.array(Images_BW)
Images_CL = np.array(Images_CL)


#Keras Imports
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Dropout, AveragePooling2D, MaxPooling2D
from keras.layers import LeakyReLU, UpSampling2D, Activation, Flatten, concatenate, BatchNormalization
from keras import backend as K

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

#Model
class GAN (object):

    def __init__ (self, steps = 1):
        
        #Generator
        self.G = None

        #Discriminator
        self.D = None

        #Old Discriminators
        self.O = None

        #Big Models
        self.DM = None
        self.AM = None
        
        #Transfer Model
        self.T = None
        
        #Steps
        self.steps = steps

        #Inputs
        self.Color = Sequential()
        self.Color.add(Dense(3, input_shape = (100, 100, 3)))

        self.Black = Sequential()
        self.Black.add(Dense(1, input_shape = (100, 100, 1)))

        #Config
        self.LR = 0.00001

    def generator (self):

        if self.G:
            self.G.compile(optimizer = Adam(lr = self.LR),
                            loss = smoothL1)
            return self.G
        

        """
        Architecture:

        x x x x x x x x x x
        |   | | | | | |   |
        |   x x x x x x   |
        |   |   | |   |   |
        |   |   x x   |   |
        |   V   | |   V   |
        |   x x x x x x   |
        V   | | | | | |   V
        x x x x x x x x x x

        """


        

        #Input
        gi = Input(shape = [3, 3, 128])
        
        gc = Conv2DTranspose(filters = 128, kernel_size = 3, padding = 'valid')(gi)
        ga = LeakyReLU()(gc)
        
        
        #DECODE
        
        #5x5
        gc = Conv2DTranspose(filters = 192, kernel_size = 5, padding = 'same')(ga)
        gu = UpSampling2D((5,5))(gc)
        ga = LeakyReLU()(gu)
        #25x25
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #25x25
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #25x25
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #50x50
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #50x50
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gb = BatchNormalization(momentum = 0.9)(ga)
        #50x50
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(gb)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #100x100
        gc = Conv2DTranspose(filters = 128, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #100x100
        gc = Conv2DTranspose(filters = 96, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #100x100
        gc = Conv2DTranspose(filters = 96, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #100x100
        gc = Conv2DTranspose(filters = 96, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #OUTPUT
        go = Dense(1, activation = 'sigmoid')(ga)

        self.G = Model(inputs = gi, outputs = go)

        self.G.compile(optimizer = Adam(lr = self.LR),
                        loss = smoothL1)

        return self.G

    def discriminator(self):
        
        if self.D:
            return self.D
        
        self.D = Sequential()
        
        #NOT PatchGAN
        #100x100
        self.D.add(Conv2D(filters = 96, kernel_size = 5, padding = 'same', input_shape = [100, 100, 1]))
        self.D.add(LeakyReLU())
        self.D.add(MaxPooling2D())
        
        #50x50
        self.D.add(Conv2D(filters = 96, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 96, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        self.D.add(MaxPooling2D())
        
        #25x25
        self.D.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 128, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 192, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        self.D.add(MaxPooling2D((5, 5)))
        
        #5x5
        self.D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        #5x5
        self.D.add(Conv2D(filters = 256, kernel_size = 3))
        self.D.add(LeakyReLU())
        
        #3x3
        self.D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        #3x3
        self.D.add(Conv2D(filters = 512, kernel_size = 3))
        self.D.add(LeakyReLU())
        
        #1x1
        #512
        self.D.add(Flatten())
        
        self.D.add(Dense(512))
        self.D.add(LeakyReLU())
        
        self.D.add(Dense(512))
        self.D.add(LeakyReLU())
        
        self.D.add(Dense(128))
        self.D.add(LeakyReLU())
        
        self.D.add(Dense(64))
        self.D.add(LeakyReLU())
        
        self.D.add(Dense(1, activation = 'sigmoid'))
        
        return self.D
        
    
    def save_old_dis(self):

        self.O = self.D.get_weights()
    
    def load_old_dis(self):

        self.D.set_weights(self.O)

    #Dis Model
    def DisModel(self):
        
        if self.DM:
            self.DM.compile(loss = 'binary_crossentropy',
                            optimizer = SGD(lr = self.LR * 4000))
            return self.DM
        
        self.DM = Sequential()

        #Add Discriminator
        self.DM.add(self.discriminator())

        #Compile
        self.DM.compile(loss = 'binary_crossentropy',
                        optimizer = SGD(lr = self.LR * 4000))

        return self.DM

        
    
    def AdModel(self):

        if self.AM:
            self.AM.compile(loss = 'binary_crossentropy',
                            optimizer = Adam(lr = self.LR))
            return self.AM

        #Gen to Smol Boi
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())

        self.AM.compile(loss = 'binary_crossentropy',
                        optimizer = Adam(lr = self.LR))
        
        return self.AM

    def TranModel(self):
        
        if self.T:
            return self.T
        
        tran_file = open("Transfer/tran.json", 'r')
        tran_json = tran_file.read()
        tran_file.close()
        
        self.T = model_from_json(tran_json)
        self.T.load_weights("Transfer/tran.h5")
        
        return self.T
    
    

    
        

class Model_GAN(object):

    def __init__(self, steps = 1):
        
        #Init GAN
        self.GAN = GAN(steps)

        #Generator
        self.generator = self.GAN.generator()

        #Init Disc
        self.GAN.discriminator()

        #Training Models
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        

        self.TranModel = self.GAN.TranModel()
    
    def train(self, batch_size = 4):
        
        self.train_dis(batch_size)
        
        if self.GAN.steps > 100:
            self.train_gen(batch_size)

        dtry = 0
        while (self.av_gen() > 0.7 or self.av_dis() < 0.2 or self.av_gen() > self.av_dis()) and dtry < 20:
            self.train_dis(batch_size)
            dtry = dtry + 1

        gtry = 0
        if self.GAN.steps > 100:
            while self.av_gen() < 0.08 and gtry < 20:
                self.train_gen(batch_size)
                gtry = gtry + 1
            
        dd = self.av_dis()
        gg = self.av_gen()
        
        print("D - " + str(dtry) + " / " + str(dd))

        print("G - " + str(gtry) + " / " + str(gg))
        
        time.sleep(0.01)
        
        #Save, Reset Memory, Evaluate
        if self.GAN.steps % 50 == 0:
            self.evaluate()
            
        if self.GAN.steps % 20 == 0:
            self.save(floor(self.GAN.steps / 200))
            self.reset_memory()

        
        
        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self, batch = 4):

        train_data = []
        label_data = []

        for i in range(int(batch/4)):
            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data.append(Images_BW[im_no])
            label_data.append(one())

            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data.append(self.generator.predict(noise(1))[0])
            label_data.append(zero())

            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data.append(Images_BW[im_no])
            label_data.append(one())

            #Fake Images
            n = random.randint(0, 4)
            im_no = random.randint(0, len(Images_BW) - 1)

            if n == 0: #Black
                train_data.append(np.zeros([100, 100, 1]))
            elif n == 1: #White
                train_data.append(np.ones([100, 100, 1]))
            elif n == 2: #Generated
                train_data.append(self.generator.predict(noise(1))[0])
            else: #Noise
                train_data.append(np.random.uniform(0.0, 1.0, size = [100, 100, 1]))



            label_data.append(zero())
        
        train_data = np.array(train_data)
        label_data = np.array(label_data)
        
        self.DisModel.train_on_batch(train_data, label_data)

    def train_gen(self, batch = 4):

        self.GAN.save_old_dis()

        
        label_data = np.ones([batch, 1])
        
        self.AdModel.train_on_batch(noise(batch), label_data)

        self.GAN.load_old_dis()
    
    def av_gen(self):

        out = self.AdModel.predict(noise(8))

        return np.sum(out) / (8.0)
    
    def av_dis(self):

        train_data = []

        for i in range(4):
            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            train_data.append(Images_BW[im_no])

            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            train_data.append(self.generator.predict(noise(1))[0])
        
        out = self.DisModel.predict(np.array(train_data))

        return np.sum(out) / (8.0)

        
        
    def convert(self):

        return self.generator.predict(noise(1))[0]
    
    def convert2(self, imput):
        
        return self.TranModel.predict([np.array([imput]), np.array([noise2()])])[0]
    
    def convert3(self):

        return self.generator.predict(np.zeros([1, 3, 3, 128]))[0]

    def evaluate(self):

        im = self.convert()
        
        im2 = self.convert()
        
        im3 = self.convert2(im2)
        
        
        
        
        plt.figure(1)
        plt.imshow(fix(im))
        
        plt.figure(2)
        plt.imshow(fix(im2))
        
        plt.figure(3)
        plt.imshow(im3)
        
        plt.show(block = False)
        plt.pause(0.01)
        
    def evaluate2(self, n = 0):

        im = self.convert2(self.convert())
        
        im2 = self.convert2(self.convert())
        
        im3 = self.convert2(self.convert())
        
        im4 = self.convert2(self.convert())
        
        im5 = self.convert2(self.convert())
        
        im6 = self.convert2(self.convert())
        
        
        show = np.concatenate([im, im2, im3], axis = 1)
        show2 = np.concatenate([im4, im5, im6], axis = 1)
        
        show3 = np.concatenate([show, show2], axis = 0)
        
        plt.figure(1)
        plt.imshow(show3)
        
        plt.show()
        
        

    def save(self, num):
        gen_json = self.GAN.G.to_json()
        dis_json = self.GAN.D.to_json()

        with open("Models/gen.json", "w") as json_file:
            json_file.write(gen_json)

        with open("Models/dis.json", "w") as json_file:
            json_file.write(dis_json)

        self.GAN.G.save_weights("Models/gen"+str(num)+".h5")
        self.GAN.D.save_weights("Models/dis"+str(num)+".h5")

        #print("Saved!")

    def load(self, num):
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Generator
        gen_file = open("Models/gen.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights("Models/gen"+str(num)+".h5")

        #Discriminator
        dis_file = open("Models/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()
        
        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights("Models/dis"+str(num)+".h5")

        #Reinitialize
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        
        self.GAN.steps = steps1
    
    def reset_memory(self):
        
        self.save('_tmp')
        steps = self.GAN.steps
        
        K.clear_session()
        
        self.__init__()
        self.load('_tmp')
        self.GAN.steps = steps



model = Model_GAN(1)
model.load(80)

#model.reset_memory()

while(True):
    #print("\n\nRound " + str(model.GAN.steps) + ":")
    
    model.evaluate2()