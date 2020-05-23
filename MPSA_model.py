#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:18:25 2020

@author: sapna
"""
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, Dropout, Lambda, Add, Reshape, \
AveragePooling2D, Average, Activation
from keras.engine.topology import Layer
import keras.backend as K
from keras_vggface.vggface import VGGFace
from keras.optimizers import *
import numpy as np
import cv2
import os
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from scipy import interp
import tensorflow as tf
from Get_Available_Gpus import get_available_gpus


class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)
    
    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1    ,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1    ,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)
    
    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape


class pan_network(object): 
    def __init__(self, classes, hidden_dim):
        self.nb_class = classes
        self.hidden_dim = hidden_dim
    def import_images(self, path):
        images = []
        for index, name in enumerate(os.listdir(path)):
            im_folder = os.path.join(path, name)
            for im in os.listdir(im_folder):
                img = cv2.imread(os.path.join(im_folder, im))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (512, 496))
                img = np.array(img).reshape(496, 512)
                img_1 = np.dstack((img, img, img))
                img = np.array(img_1).reshape(496, 512, 3)
                img = cv2.resize(img, (112,112))
                img = np.array(img).reshape(112, 112, 3)
                if img is not None:
                    images.append((np.array(img), index))
        return images
    
    def import_maps(self, path):
       images = []
       for index, name in enumerate(os.listdir(path)):
           im_folder = os.path.join(path, name)
           for im in os.listdir(im_folder):
                img = cv2.imread(os.path.join(im_folder, im))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (56, 56))
                if img is not None:
                    images.append(np.array(img))
       return images  
       
    def MPSA(self, w=28, h=28, c=256, dout=512):    
        v1 = Input(shape = (w,h,c))
        q1 = Input(shape = (w*2,h*2,c//2))
        k1 = Input(shape = (w*2,h*2,c//2))
        att_map1 = Input(shape = (w*2,h*2,1)) 
        k2 = Add()([k1, att_map1])
        k3 = AveragePooling2D()(k2)
        q2 = Add()([q1, att_map1])
        q3 = AveragePooling2D()(q2)
        v = Reshape([w*h,256])(v1)
        q = Reshape([w*h,128])(q3)
        k = Reshape([w*h,128])(k3)   
        att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]),
                     output_shape=(w*h,w*h))([q,k]) # 49*49
        att = Lambda(lambda x:  K.softmax(x), output_shape=(w*h,w*h))(att)
        out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]),  \
                     output_shape=(w*h,256))([att,v])
        out_2 = Reshape([w,h,256])(out)
        out_1 = Add()([out_2, v1])
        return  Model(inputs=[v1,q1,k1,att_map1], outputs=out_1)
    
    def MDA(self, w=14, h=14, c=512):
        v1 = Input(shape = (w,h,c))
        q1 = Input(shape = (w,h,c))
        k1 = Input(shape = (w,h,c))
        v = Reshape([w*h,512])(v1)
        q = Reshape([w*h,512])(q1)
        k = Reshape([w*h,512])(k1)
        att= Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,1]),
                     output_shape=(512,512))([q,k])
        out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]), \
                     output_shape=(w*h,512))([v,att])   
        out = Reshape([w,h,512])(out)
        out = Add()([out, v1])
        return  Model(inputs=[v1,q1,k1], outputs=out)
    

    def model(self):
        att_img = Input(shape=(56, 56, 1))
        vgg_model = VGGFace(include_top=False, input_shape=(112, 112, 3))
        vgg_model.summary()
        x1 = vgg_model.get_layer('conv2_1').output
        x2 = vgg_model.get_layer('conv2_2').output
        x3 = vgg_model.get_layer('conv3_3').output       
        att_1 = self.MPSA()
        x_1 = att_1([x3,x2,x1, att_img])
        x_1 = Activation('relu')(x_1)
        x_1 = NormL()(x_1)
        x_1 =   AveragePooling2D()(x_1)
        x_1 = Dense(32, activation='relu')(x_1)
        x = Flatten()(x_1) 
        x = Dense(self.hidden_dim, activation='relu', name='fc6')(x)
        x = Dropout(0.25)(x)
        x = Dense(self.hidden_dim, activation='relu', name='fc7')(x)
        x = Dropout(0.25)(x)
        out = Dense(self.nb_class, activation='softmax', name='fc8')(x)
        custom_vgg_model = Model([vgg_model.input, att_img], out)
        return custom_vgg_model
    
    
gpus = get_available_gpus(1)
with tf.device(gpus[0]): 
    cvscores = []
    cvp = []
    cvr = []
    batch_size = 32
    PAN = pan_network(classes = 4, hidden_dim =512)
    final_model = PAN.model()
    final_model.summary()
    opt =SGD(lr=0.001)
    final_model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])

    atten_maps_1 = PAN.import_maps('/home/bappaditya/Sapna/OCT2017_attmap/train')
    atten_maps_2 = PAN.import_maps('/home/bappaditya/Sapna/OCT2017_attmap/val')
    atten_maps_3 = PAN.import_maps('/home/bappaditya/Sapna/OCT2017_attmap/test')
    atten_maps_l =  atten_maps_1 + atten_maps_2 + atten_maps_3
    atten_maps =  np.array(atten_maps_l).reshape(len(atten_maps_l),56,56,1)
    images_labels_1 = PAN.import_images('/home/bappaditya/Sapna/OCT2017/train')
    images_labels_2 = PAN.import_images('/home/bappaditya/Sapna/OCT2017/val')
    images_labels_3 = PAN.import_images('/home/bappaditya/Sapna/OCT2017/test')
    images_labels = images_labels_1 + images_labels_2 + images_labels_3
    
	
    train_labels = [i[1] for i in images_labels]
    train_labels = np.array(train_labels)
	
    tr_img_data_l = list([i[0] for i in images_labels])
    tr_img_data = np.array(tr_img_data_l)
    min_max_scalar = MinMaxScaler(copy = False)
    min_max_scalar.partial_fit(tr_img_data.reshape(len(tr_img_data_l), 112 * 112 * 3))
    tr_img = min_max_scalar.transform(tr_img_data.reshape(len(tr_img_data_l), 112 * 112 * 3))
    tr_img_data = tr_img.reshape(len(tr_img_data_l), 112, 112, 3)		

    min_max_scalar_att = MinMaxScaler(copy = False)
    min_max_scalar_att.partial_fit(atten_maps.reshape(len(atten_maps_l), 56 * 56 * 1))
    att_maps = min_max_scalar_att.transform(atten_maps.reshape(len(atten_maps_l), 56 * 56 * 1))
    atten_maps = att_maps.reshape(len(atten_maps_l), 56, 56, 1)


    gen = ImageDataGenerator(
        width_shift_range=40,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
	
    testgen = ImageDataGenerator()

    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_acc', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4,
                                           mode='min')
        return [mcp_save, reduce_lr_loss]
	
    folds = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=32).split(tr_img_data, train_labels))	
	
    for j, (train_idx, val_idx) in enumerate(folds):
        X_train_cv = tr_img_data[train_idx]
        X_train_att_cv = atten_maps[train_idx]
        y_train_cv_1 = train_labels[train_idx]
        train_labels_encoded = OneHotEncoder()
        y_train_cv = train_labels_encoded.fit_transform(y_train_cv_1.reshape(-1, 1)).toarray()

        X_valid_cv = tr_img_data[val_idx]
        X_valid_att_cv = atten_maps[val_idx]
        y_valid_cv_1 = train_labels[val_idx]
        train_labels_encoded = OneHotEncoder()
        y_valid_cv = train_labels_encoded.fit_transform(y_valid_cv_1.reshape(-1, 1)).toarray()

        name_weights = "final_model_oct17_eccvMPSA_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)

        def generator_two_img(X1, X2, y, batch_size):
            genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
            genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
            while True:
                X1i = genX1.next()
                X2i = genX2.next()
                yield [X1i[0], X2i[0]], X1i[1]

        def valid_two_img(X1, X2, y, batch_size):
            genX1 = testgen.flow(X1, y,  batch_size=batch_size, seed=1)
            genX2 = testgen.flow(X2, y, batch_size=batch_size, seed=1)
            while True:
                X1i = genX1.next()
                X2i = genX2.next()
                yield [X1i[0], X2i[0]], X1i[1]

      #  generator = gen.flow([X_valid_cv,X_valid_att_cv], y_valid_cv, batch_size=batch_size)
        final_model.fit_generator(
            generator_two_img(X_valid_cv, X_valid_att_cv, y_valid_cv, batch_size = batch_size),
            steps_per_epoch=len(X_valid_cv//batch_size),
            epochs=10,
            shuffle=True,
            verbose=2,
            validation_data = valid_two_img(X_train_cv, X_train_att_cv, y_train_cv, batch_size = batch_size),
            validation_steps=len(X_train_cv//batch_size),
            callbacks=callbacks
        )
        
        score = final_model.evaluate([X_train_cv, X_train_att_cv], y_train_cv, \
                                     batch_size=batch_size, verbose = 2)
        print("Accuracy = " + format(score[1] * 100, '.2f') + "%")
        y_predict = final_model.predict([X_train_cv, X_train_att_cv], \
                                        batch_size=batch_size)
        y_pred = np.argmax(y_predict, axis=1)
        y_true = y_train_cv_1
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        cc = confusion_matrix(y_true, y_pred)
        print("Precision = ", precision),
        print("Recall = ", recall)
        print("Confusion Matrix", cc)
        cvscores.append(score[1] * 100)
        cvp.append(precision * 100)
        cvr.append(recall * 100)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(y_train_cv[:, i], y_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        n_classes = 4
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        print("auc=", roc_auc["macro"])
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_eccvMPSA_oct17_fpr.npy'.format(j), all_fpr)
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_eccvMPSA_oct17_tpr.npy'.format(j), mean_tpr)
        final_model.save('/home/bappaditya/Sapna/Codes/fold_{0}_eccvMPSA_oct17.hdf5'.format(j))

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvp), np.std(cvp)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvr), np.std(cvr)))