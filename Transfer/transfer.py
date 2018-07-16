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

def noise(col = -1):
    if col >= 0:
        output = np.concatenate((
                top_colors(Images_CL[col]),
                np.random.uniform(0.0, 1.0, size = [119])
                ))
    else:
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


for n in range(0, 151):
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
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Dropout, AveragePooling2D
from keras.layers import LeakyReLU, UpSampling2D, Activation, Merge, Flatten, concatenate, BatchNormalization
from keras import backend as K

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)

#Model
class GAN (object):

    def __init__ (self):
        
        #Generator
        self.G = None

        #Discriminator
        self.D = None
        self.DB = None

        #Old Discriminators
        self.O = None

        #Big Models
        self.DM = None
        self.AM = None

        #Inputs
        self.Color = Sequential()
        self.Color.add(Dense(3, input_shape = (100, 100, 3)))

        self.Black = Sequential()
        self.Black.add(Dense(1, input_shape = (100, 100, 1)))

        #Config
        self.LR = 0.0003

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
        gi = Input(shape = [100, 100, 1])
        gn = Input(shape = [5, 5, 128])
        
        #100x100
        gc = Conv2D(filters = 32, kernel_size = 3, padding = 'same')(gi)
        gp = AveragePooling2D()(gc)
        ga = LeakyReLU()(gp)
        gd = Dropout(0.2)(ga)
        #50x50
        gt1 = Activation('linear')(gd)
        gc = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #50x50
        gc = Conv2D(filters = 96, kernel_size = 3, padding = 'same')(ga)
        gp = AveragePooling2D()(gc)
        ga = LeakyReLU()(gp)
        #25x25
        gt2 = Activation('linear')(ga)
        gc = Conv2D(filters = 96, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #25x25
        gc = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(ga)
        gp = AveragePooling2D((5, 5))(gc)
        ga = LeakyReLU()(gp)
        #5x5
        
        #ENCODED
        
        
        gc = Conv2D(filters = 128, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        
        #Noise
        gi2 = Input(shape = [5, 5, 128])
        gn = concatenate([ga, gi2])
        
        gc = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(gn)
        ga = LeakyReLU()(gc)
        
        gc = Conv2D(filters = 256, kernel_size = 3, padding = 'same')(gn)
        ga = LeakyReLU()(gc)
        
        
        #DECODE
        
        #5x5
        gc = Conv2DTranspose(filters = 192, kernel_size = 5, padding = 'same')(ga)
        gu = UpSampling2D((5,5))(gc)
        ga = LeakyReLU()(gu)
        #25x25
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt2])
        #25x25
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
        ga = LeakyReLU()(gc)
        #25x25
        gc = Conv2D(filters = 192, kernel_size = 3, padding = 'same')(ga)
        gu = UpSampling2D()(gc)
        ga = LeakyReLU()(gu)
        #50x50
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        gm = concatenate([ga, gt1])
        #50x50
        gc = Conv2DTranspose(filters = 192, kernel_size = 3, padding = 'same')(gm)
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
        gc = Conv2DTranspose(filters = 64, kernel_size = 3, padding = 'same')(ga)
        ga = LeakyReLU()(gc)
        #OUTPUT
        go = Dense(3, activation = 'sigmoid')(ga)

        self.G = Model(inputs = [gi, gi2], outputs = go)

        self.G.compile(optimizer = Adam(lr = self.LR),
                        loss = smoothL1)

        return self.G

    def discriminator_small (self):

        #If it exists, return
        if self.D:
            return self.D

        #MERGE BEFORE
        self.D = Sequential()

        #PatchGAN
        #100x100 -> 20x20
        self.D.add(Conv2D(filters = 2048, kernel_size = 5, padding = 'valid', strides = 5, input_shape = [100, 100, 4]))
        self.D.add(LeakyReLU())

        #Individual Pixel Neural Networks Pretty Much
        self.D.add(Conv2D(filters = 512, kernel_size = 1))
        self.D.add(LeakyReLU())
        self.D.add(Conv2D(filters = 512, kernel_size = 1))
        self.D.add(LeakyReLU())
        self.D.add(Conv2D(filters = 128, kernel_size = 1))
        self.D.add(LeakyReLU())
        self.D.add(Conv2D(filters = 1, kernel_size = 1))
        self.D.add(LeakyReLU())

        return self.D
    
    def discriminator(self):
        
        if self.D:
            return self.D
        
        self.D = Sequential()
        
        #Not PatchGAN
        #100x100
        self.D.add(Conv2D(filters = 96, kernel_size = 5, padding = 'same', input_shape = [100, 100, 4]))
        self.D.add(LeakyReLU())
        self.D.add(AveragePooling2D())
        
        #50x50
        self.D.add(Conv2D(filters = 96, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 96, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        self.D.add(AveragePooling2D())
        
        #25x25
        self.D.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 128, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 192, kernel_size = 5, padding = 'same'))
        self.D.add(LeakyReLU())
        self.D.add(AveragePooling2D((5, 5)))
        
        #5x5
        self.D.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same'))
        self.D.add(LeakyReLU())
        
        self.D.add(Conv2D(filters = 256, kernel_size = 3))
        self.D.add(LeakyReLU())
        
        #3x3
        self.D.add(Conv2D(filters = 512, kernel_size = 3))
        self.D.add(LeakyReLU())
        
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
                            optimizer = Adam(lr = self.LR))
            return self.DM
        
        self.DM = Sequential()

        #Merge Big and Small
        self.DM.add(Merge([self.Color, self.Black], mode='concat'))

        #Add Discriminator
        self.DM.add(self.D)

        #Compile
        self.DM.compile(loss = 'binary_crossentropy',
                        optimizer = Adam(lr = self.LR))

        return self.DM

        
    
    def AdModel(self):

        if self.AM:
            self.AM.compile(loss = 'binary_crossentropy',
                            optimizer = Adam(lr = self.LR))
            return self.AM

        #Gen to Smol Boi
        self.AM = Sequential()
        self.AM.add(Merge([self.G, self.Black], mode='concat'))
        self.AM.add(self.D)

        self.AM.compile(loss = 'binary_crossentropy',
                        optimizer = Adam(lr = self.LR))
        
        return self.AM



    
        

class Model_GAN(object):

    def __init__(self, iterations = 1):
        
        #Init GAN
        self.GAN = GAN()

        #Generator
        self.generator = self.GAN.generator()

        #Init Disc
        #self.GAN.discriminator()

        #Training Models
        #self.DisModel = self.GAN.DisModel()
        #self.AdModel = self.GAN.AdModel()
        
        self.genny = []
        #self.dizzy = []



        #self.generator.summary()
        #self.GAN.D.summary()

        self.iterations = iterations
        self.loss = []
    
    def train_encoder(self, batch = 4):

        train_data = []
        train_data2 = []
        label_data = []

        for i in range(batch):

            im_no = random.randint(0, len(Images_BW) - 1)
            train_data.append(Images_BW[im_no])
            train_data2.append(noise(im_no))
            label_data.append(Images_CL[im_no])

        return self.generator.train_on_batch([np.array(train_data), np.array(train_data2)], np.array(label_data))

    def train(self, batch_size = 4):
        
        #self.train_encoder(batch_size)
        loss = self.train_encoder(batch_size)

        #self.train_dis(batch_size)

        #self.train_gen(batch_size)

        #dtry = 0
        #while (self.av_gen() > 0.7 or self.av_dis() < 0.2 or self.av_gen() > self.av_dis()) and dtry < 20:
        #    self.train_dis(batch_size)
        #    dtry = dtry + 1

        #gtry = 0
        #while self.av_gen() < 0.1 and gtry < 20:
        #    self.train_encoder(batch_size)
        #    self.train_gen(batch_size)
        #    gtry = gtry + 1
            
        #dd = self.av_dis()
        #gg = self.av_gen()
        
        #print("D - " + str(dtry) + " / " + str(dd))

        #print("G - " + str(gtry) + " / " + str(gg))
        
        #self.dizzy.append(dd)
        self.loss.append(loss)
        
        time.sleep(0.1)
        
        print(loss)
        
        #Save, Reset Memory, Evaluate
        if self.iterations % 50 == 0:
            self.evaluate()
            
        if self.iterations % 10 == 0:
            self.save(floor(self.iterations / 200))
            self.reset_memory()

        
        
        self.iterations = self.iterations + 1

    def train_dis(self, batch = 4):

        train_data1 = []
        train_data2 = []
        label_data = []

        for i in range(int(batch/4)):
            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data1.append(Images_CL[im_no])
            train_data2.append(Images_BW[im_no])
            label_data.append(one())

            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data1.append(self.generator.predict([np.array([Images_BW[im_no]]), noise(1)])[0])
            train_data2.append(Images_BW[im_no])
            label_data.append(zero())

            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data1.append(Images_CL[im_no])
            train_data2.append(Images_BW[im_no])
            label_data.append(one())

            #Fake Images
            n = random.randint(0, 4)
            im_no = random.randint(0, len(Images_BW) - 1)

            if n == 0: #Black
                train_data1.append(np.zeros([100, 100, 3]))
            elif n == 1: #White
                train_data1.append(np.ones([100, 100, 3]))
            elif n == 2: #Noise
                train_data1.append(np.random.uniform(0, 1.0, size = [100, 100, 3]))
            elif n == 3: #Generated
                train_data1.append(self.generator.predict([np.array([Images_BW[im_no]]), noise(1)])[0])
            elif n == 4: #Any Given Color
                arr = np.zeros([100, 100, 3])

                for j in range(3):
                    if random.random() > 0.5:
                        for r in range(100):
                            for c in range(100):
                                arr[r][c][j] = 1
                
                train_data1.append(arr)



            train_data2.append(Images_BW[im_no])
            label_data.append(zero())
        
        train_data1 = np.array(train_data1)
        train_data2 = np.array(train_data2)
        label_data = np.array(label_data)
        
        self.DisModel.train_on_batch([train_data1, train_data2], label_data)

    def train_gen(self, batch = 4):

        self.GAN.save_old_dis()

        train_data1 = []
        train_data2 = []
        label_data = []

        for i in range(batch):
            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            
            train_data1.append(Images_BW[im_no])
            train_data2.append(Images_BW[im_no])
            label_data.append(one())
        
        self.AdModel.train_on_batch([np.array(train_data1),
                                     noise(batch),
                                    np.array(train_data2)],
                                    np.array(label_data))

        self.GAN.load_old_dis()
    
    def av_gen(self):

        train_data1 = []
        train_data2 = []

        for i in range(8):
            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            train_data1.append(Images_BW[im_no])
            train_data2.append(Images_BW[im_no])
        
        out = self.AdModel.predict([np.array(train_data1),
                                    noise(8),
                                    np.array(train_data2)])

        return np.sum(out) / (8.0)
    
    def av_dis(self):

        train_data1 = []
        train_data2 = []

        for i in range(4):
            #Real Image
            im_no = random.randint(0, len(Images_BW) - 1)
            train_data1.append(Images_CL[im_no])
            train_data2.append(Images_BW[im_no])

            #Fake Images
            im_no = random.randint(0, len(Images_BW) - 1)
            train_data1.append(self.generator.predict([np.array([Images_BW[im_no]]), noise(1)])[0])
            train_data2.append(Images_BW[im_no])
        
        out = self.DisModel.predict([np.array(train_data1), np.array(train_data2)])

        return np.sum(out) / (8.0)

        
        
    def convert(self, image_no):

        return self.generator.predict([np.array([Images_BW[image_no]]), np.array([noise()])])[0]
    
    def convert2(self, image_no):

        return self.generator.predict([np.array([Images_BW[image_no]]), np.array([noise(image_no)])])[0]

    def evaluate(self):

        n = random.randint(0, len(Images_BW)-1)

        plt.figure(1)
        plt.imshow(fix(Images_BW[n]))
        
        plt.figure(2)
        #plt.imshow(Images_CL[n])
        
        im = self.convert2(n)
        
        im2 = self.convert(n)
        
        
        
        
        plt.figure(3)
        plt.imshow(im)
        
        plt.figure(4)
        plt.imshow(im2)
        
        
        
        plt.show(block = False)
        plt.pause(0.01)
        
        

    def save(self, num):
        gen_json = self.GAN.G.to_json()
        #dis_json = self.GAN.D.to_json()

        with open("Models/gen.json", "w") as json_file:
            json_file.write(gen_json)

        #with open("Models/dis.json", "w") as json_file:
        #    json_file.write(dis_json)

        self.GAN.G.save_weights("Models/gen"+str(num)+".h5")
        #self.GAN.D.save_weights("Models/dis"+str(num)+".h5")

        #print("Saved!")

    def load(self, num):
        self.GAN = None
        self.GAN = GAN()

        #Generator
        gen_file = open("Models/gen.json", 'r')
        gen_json = gen_file.read()
        gen_file.close()
        
        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights("Models/gen"+str(num)+".h5")

        #Discriminator
        #dis_file = open("Models/dis.json", 'r')
        #dis_json = dis_file.read()
        #dis_file.close()
        
        #self.GAN.D = model_from_json(dis_json)
        #self.GAN.D.load_weights("Models/dis"+str(num)+".h5")

        #Reinitialize
        #self.generator = self.GAN.generator()
        #self.DisModel = self.GAN.DisModel()
        #self.AdModel = self.GAN.AdModel()
    
    def reset_memory(self):
        
        self.save('_tmp')
        steps = self.iterations
        
        K.clear_session()
        
        self.__init__()
        self.iterations = steps
        self.load('_tmp')
        self.generator = self.GAN.generator()



model = Model_GAN(0)

#model.GAN.G.summary()

model.load(283)
model.reset_memory()

while(True):
    print("\n\nRound " + str(model.iterations))
    
    model.evaluate()
    
    if model.iterations % 200 == 1:
        print("Assigning New Hues!")
        for i in range(len(Images_CL)):
            Images_CL[i] = process(Images_CL[i])