from tkinter import *
from tkinter.ttk import *
import tkinter
import keras
from PIL import Image,ImageTk
import cv2
import numpy as np
import random
import tensorflow as tf

model = keras.models.load_model("D:\\20202\\Project2\\CNN\\my_model_CNN")
model_DNN = keras.models.load_model("D:\\20202\\Project2\\Project2_ML\\my_model")
window = Tk()
window.title("Nhận diện số viết tay")
window.geometry("1020x500")

nameProgram = tkinter.Label(window,text = "Nhận diện số viết tay")
nameProgram.config(font=("Arial", 30))
nameProgram.pack(pady=10)

# Load dữ liệu MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train = X_train[:50000, :], y_train[:50000]


def inputImage():
    val = random.randint(0, 9999)
    dim = (327, 327)
    resized = cv2.resize(X_test[val], dim, interpolation=cv2.INTER_AREA)
    img = Image.fromarray(resized)
    imgtk = ImageTk.PhotoImage(image=img)
    choseImage.configure(image = imgtk)
    choseImage.image = imgtk
    # CNN
    y_predict = model.predict(X_test[val].reshape(1,28,28,1))
    index = np.argmax(y_predict)
    p = y_predict[0][index] * 100
    p = round(p, 4)
    RStxt_CNN.configure(text="Số dự đoán: " + str(index))
    Plabel_CNN.configure(text="Tỷ lệ chính xác:" + str(p) + " %")

    # DNN
    y_predict_DNN = model_DNN.predict(X_test[val].reshape(1, 28 * 28))
    indexDNN = np.argmax(y_predict_DNN)
    pDNN = y_predict_DNN[0][indexDNN] * 100
    pDNN = round(pDNN,4)
    RStxt_DNN.configure(text="Số dự đoán: " + str(indexDNN))
    Plabel_DNN.configure(text="Tỷ lệ chính xác:" + str(pDNN) + " %")
    return


load = Image.open("D:\\20202\\Project2\\CNN\\CNNProject\\white.JPG")
render = ImageTk.PhotoImage(load)


# anhmau
anhmau = tkinter.Label(window,text = "Ảnh Nhận Diện")
anhmau.config(font=("Arial", 15))
anhmau.place(x=150, y = 70)

choseImage = tkinter.Label(window,image=render)
choseImage.place(x= 50, y = 100)

btnChose = Button(window, text = "Random ảnh", command = inputImage)
btnChose.place(x = 165 , y =440)

DNNLabel = tkinter.Label(window,text = "DNN")
DNNLabel.config(font=("Arial", 25))
DNNLabel.place(x= 520, y = 150)

CNNLabel = tkinter.Label(window,text = "CNN")
CNNLabel.config(font=("Arial", 25))
CNNLabel.place(x= 800, y = 150)

RStxt_DNN = tkinter.Label(window,text = "Số dự đoán: ")
RStxt_DNN.config(font=("Arial", 25))
RStxt_DNN.place(x= 480, y = 225)

Plabel_DNN = tkinter.Label(window,text = "Tỷ lệ chính xác:")
Plabel_DNN.config(font = ("Arial", 15))
Plabel_DNN.place(x= 480, y = 350)

RStxt_CNN = tkinter.Label(window,text = "Số dự đoán: ")
RStxt_CNN.config(font=("Arial", 25))
RStxt_CNN.place(x= 750, y = 225)

Plabel_CNN = tkinter.Label(window,text = "Tỷ lệ chính xác:")
Plabel_CNN.config(font = ("Arial", 15))
Plabel_CNN.place(x= 750, y = 350)

model.predict(X_test[1].reshape(1,28,28,1))
model_DNN.predict(X_test[1].reshape(-1, 28 * 28))

window.mainloop()



