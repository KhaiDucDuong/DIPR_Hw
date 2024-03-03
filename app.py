import cv2
import tkinter
from tkinter import filedialog as fd
import numpy as np
import PIL.Image, PIL.ImageTk

class App:
    def __init__(self, window, window_title, image_path="assets/95.jpg"):
        self.window = window
        self.window.title(window_title)
        screenWidth= self.window.winfo_screenwidth()               
        screenHeight= self.window.winfo_screenheight()               
        self.window.geometry("%dx%d" % (screenWidth, screenHeight))
 
        # Load an image using OpenCV
        self.cv_img = cv2.imread(image_path, 0)
 
         # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width = self.cv_img.shape
 
        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width = screenWidth / 2 - 50, height = screenHeight / 2 - 50)
        self.canvas.pack()
 
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
 
        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.scaler1 = tkinter.Scale(window, from_=0, to=5,length=800, resolution=0.1, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler1.set(1)
        self.scaler1.pack()
        self.scaler2 = tkinter.Scale(window, from_=0, to=100,length=800, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler2.set(50)
        self.scaler2.pack()

        self.btn_select_img=tkinter.Button(window, text="Select Img", width=50,command=self.select_image)
        self.btn_select_img.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_filter=tkinter.Button(window, text="Filter", width=50,command=self.filter)
        self.btn_filter.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_lowpass_filter=tkinter.Button(window, text="Low-pass Filter", width=50,command=self.lowpass_filter)
        self.btn_lowpass_filter.pack(anchor=tkinter.CENTER, expand=True)

        self.btn_highpass_butterworth=tkinter.Button(window, text="High-pass Butterworth", width=50,command=self.highpass_butterworth)
        self.btn_highpass_butterworth.pack(anchor=tkinter.CENTER, expand=True)

 
        self.window.mainloop()

    def select_image(self):
        filename = fd.askopenfilename()
        if filename:
            self.image_path = filename
            # Load an image using OpenCV
            self.cv_img = cv2.imread(self.image_path, 0)

            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
    
            # Add a PhotoImage to the Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def filter(self):
        #Filtering
        F = np.fft.fft2(self.cv_img)
        F = np.fft.fftshift(F)
        M, N = self.cv_img.shape
        D0 = self.scaler2.get()

        u = np.arange(0, M)-M/2
        v=np.arange(0, N) - N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D<=D0, 'float')
        G = H*F

        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        self.cv_img = imgOut

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
    
    def lowpass_filter(self):
        F = np.fft.fft2(self.cv_img)
        n= self.scaler1.get()
        D0= self.scaler2.get()

        F=np.fft.fftshift(F)
        M, N = self.cv_img.shape
        u = np.arange(0, M)-M/2
        v = np.arange(0,N)-N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U,2) + np.power(V,2))
        H = 1/np.power(1+ (D/D0), (2*n))
        G = H*F
        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        self.cv_img = imgOut

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def highpass_butterworth(self):
        F = np.fft.fft2(self.cv_img)
        M, N = self.cv_img.shape
        n = self.scaler1.get()
        D0 = self.scaler2.get()
        u = np.arange(0, M, 1)
        v = np.arange(0,N,1)
        idx = (u > M/2)
        u[idx] = u[idx] - M
        idy = (v > N/2)
        v[idy] = v[idy] - N
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = 1/np.power(1 + (D0/(D+1e-10)), 2*n)
        G = H * F
        imgOut = np.real(np.fft.ifft2(G))
        self.cv_img = imgOut

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def highpass_ideal(self):
        pass

    def highpass_gaussian(self):
        pass
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")