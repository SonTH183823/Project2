from tkinter import *
from tkinter.ttk import *
import tkinter
from tkinter import filedialog
import keras
from PIL import Image,ImageTk
import cv2
import numpy as np
import random
import tensorflow as tf


window = Tk()
window.title("Nhận diện số viết tay")
window.geometry("800x500")

nameProgram = tkinter.Label(window,text = "Nhận diện số viết tay")
nameProgram.config(font=("Arial", 30))
nameProgram.pack(pady=10)



# 2. Load dữ liệu MNIST
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
    index = np.argmax(model.predict(X_test[val].reshape(-1, 28 * 28)))
    RStxt.configure(text="Số dự đoán: " + str(index))
    return

# def checkBtn():
#     RStxt.configure(text="Số dự đoán: "+str(index))
#     return

load = Image.open("D:\\20202\\Project2\\Project2_ML\\pythonProject\\white.JPG")
render = ImageTk.PhotoImage(load)


# anhmau
anhmau = tkinter.Label(window,text = "Ảnh Nhận Diện")
anhmau.config(font=("Arial", 15))
anhmau.place(x=150, y = 70)

choseImage = tkinter.Label(window,image=render)
choseImage.place(x= 50, y = 100)

btnChose = Button(window, text = "Random ảnh", command = inputImage)
btnChose.place(x = 165 , y =440)

# # kiểm tra
# btnCheck = Button(window,text = "Check", command =  checkBtn)
# btnCheck.place(x = 380, y = 350)

RStxt = tkinter.Label(window,text = "Số dự đoán: ")
RStxt.config(font=("Arial", 25))
RStxt.place(x= 480, y = 225)

# Ptxt = tkinter.Label(window,text = "99%")
# Ptxt.config(font=("Arial", 20))
# Ptxt.place(x= 500, y = 345)
model = keras.models.load_model("D:\\20202\\Project2\\Project2_ML\\my_model")

window.mainloop()



