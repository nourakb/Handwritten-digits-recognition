
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import torch
from torch.autograd.grad_mode import F 
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from skimage.util import view_as_blocks
from sklearn.linear_model import LinearRegression
from numpy import asarray
from PIL import Image, ImageOps
import matplotlib.pyplot as plt



def slice(image):
    block_shape = (7, 7)
    cells= view_as_blocks(image.numpy(), block_shape)
    flatten_cells = cells.reshape((16,7,7))
    return flatten_cells

def extract_features(image):
    extracted_image= np.array([[0,0,0]])
    N=np.count_nonzero(image)
    for cell in image:
        #f1
        n=np.count_nonzero(cell)
        f1=n/N
        #f2 et f3
        X= np.nonzero(cell)[1].reshape(-1, 1)
        Y= np.nonzero(cell)[0]
        if(X.size != 0):
            linear_regressor = LinearRegression()  # create object for the class
            linear_regressor.fit(X, Y)  # perform linear regression
            b=linear_regressor.coef_[0]
            f2=(2*b)/(1+pow(b,2))
            f3=(1-pow(b,2))/(1+pow(b,2))
        else : 
            f2=0
            f3=0
        features=[f1,f2,f3] 
        extracted_image=np.append(extracted_image,[features], axis=0)
    extracted_image=np.delete(extracted_image,0,axis=0)
    return extracted_image

model =torch.load('mymodel')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img.save("circles.png", "png")#juste pour tester le resultat obtenu
    #on inverse l'image pour qu'elle soit compatible au donne du mnist
    img=ImageOps.invert(img)
    print(np.array(img))
    img =torch.from_numpy(np.array(img))
    img = slice(img)#on decoupe notre image en 16 morceaux
    img=extract_features(img)#on applique l'algorithme du features extraction
    img =torch.from_numpy(np.array(img))
    #predicting the outputs
    with torch.no_grad():   
        outputs=model(img.reshape(1,48).float())
    return np.argmax(np.array(outputs)) , max(outputs)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="ecrivez un numéro", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "...............................Prédiction.................................", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2 )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND) 
        im = ImageGrab.grab(rect)
        
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=5
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
app.mainloop()