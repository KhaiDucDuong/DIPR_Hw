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
        self.img_width = screenWidth / 2 - 50
        self.img_height = screenHeight / 2 - 50
        self.scaler_width = screenWidth / 2 - 50
        self.button_width = 50
        print(int(self.img_width))
        # Load an image using OpenCV
        self.image_path = image_path
        self.cv_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.modified_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)
        self.modified_img = cv2.resize(self.cv_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape
 
        # create 2 canvases
        self.canvas_original_img = tkinter.Canvas(window, width = self.img_width, height = self.img_height)
        self.canvas_original_img.grid(column=0, row=0, rowspan=5, columnspan=2, padx=5, pady=5)

        self.canvas_modified_img = tkinter.Canvas(window, width = self.img_width, height = self.img_height)
        self.canvas_modified_img.grid(column=0, row=5, rowspan=5, columnspan=2, padx=5, pady=5)
 
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
 
        # Add a PhotoImage to the Canvas
        self.canvas_original_img.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)


        self.scaler1 = tkinter.Scale(window, from_=0, to=5,length=self.scaler_width, resolution=0.1, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler1.set(1)
        self.scaler1.grid(column=2, row=0, columnspan=2, padx=5, pady=5, sticky=tkinter.E)
        self.scaler2 = tkinter.Scale(window, from_=0, to=100,length=self.scaler_width, tickinterval=10, orient=tkinter.HORIZONTAL)
        self.scaler2.set(50)
        self.scaler2.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky=tkinter.E)
        self.scaler3 = tkinter.Scale(window, from_=0, to=50,length=self.scaler_width, tickinterval=5, orient=tkinter.HORIZONTAL)
        self.scaler3.set(10)
        self.scaler3.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky=tkinter.E)

        self.btn_select_img=tkinter.Button(window, text="Select Img", width=self.button_width,command=self.select_image)
        self.btn_select_img.grid(column=2, row=3, columnspan=1, padx=5, pady=5)

        self.btn_reset_img=tkinter.Button(window, text="Reset Img", width=self.button_width,command=self.reset_image)
        self.btn_reset_img.grid(column=3, row=3, columnspan=1, padx=5, pady=5)

        self.btn_filter=tkinter.Button(window, text="Filter", width=self.button_width,command=self.filter)
        self.btn_filter.grid(column=2, row=4, columnspan=2, padx=5, pady=5)

        self.btn_lowpass_filter=tkinter.Button(window, text="Low-pass Filter", width=self.button_width,command=self.lowpass_filter_color)
        self.btn_lowpass_filter.grid(column=2, row=5, columnspan=2, padx=5, pady=5)

        self.btn_highpass_butterworth=tkinter.Button(window, text="High-pass Butterworth", width=self.button_width,command=self.highpass_butterworth_color)
        self.btn_highpass_butterworth.grid(column=2, row=6, columnspan=1, padx=5, pady=5)

        self.btn_highpass_ideal=tkinter.Button(window, text="High-pass Ideal", width=self.button_width,command=self.highpass_ideal_color)
        self.btn_highpass_ideal.grid(column=3, row=6, columnspan=1, padx=5, pady=5)

        self.btn_erode=tkinter.Button(window, text="Erode Img", width=self.button_width,command=self.erode_img)
        self.btn_erode.grid(column=2, row=7, columnspan=1, padx=5, pady=5)

        self.btn_dilate=tkinter.Button(window, text="Dilate Img", width=self.button_width,command=self.dilate_img)
        self.btn_dilate.grid(column=3, row=7, columnspan=1, padx=5, pady=5)

        self.btn_opening=tkinter.Button(window, text="Openning Img", width=self.button_width,command=self.opening_img)
        self.btn_opening.grid(column=2, row=8, columnspan=1, padx=5, pady=5)

        self.btn_closing=tkinter.Button(window, text="Closing Img", width=self.button_width,command=self.closing_img)
        self.btn_closing.grid(column=3, row=8, columnspan=1, padx=5, pady=5)

 
        self.window.mainloop()

    def select_image(self):
        filename = fd.askopenfilename()
        if filename:
            self.image_path = filename
            # Load an image using OpenCV
            self.cv_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            self.modified_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            self.cv_img = cv2.resize(self.cv_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)
            self.modified_img = cv2.resize(self.cv_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)

            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))

            # Add a PhotoImage to the Canvas
            self.canvas_original_img.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def reset_image(self):
        self.modified_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.modified_img = cv2.resize(self.cv_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))

        # Add a PhotoImage to the Canvas
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def filter(self):
        #Filtering
        F = np.fft.fft2(self.modified_img)
        F = np.fft.fftshift(F)
        M, N = self.modified_img.shape
        D0 = self.scaler2.get()

        u = np.arange(0, M)-M/2
        v=np.arange(0, N) - N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        H = np.array(D<=D0, 'float')
        G = H*F

        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        self.modified_img = imgOut

        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)
    
    def lowpass_filter_color(self):
        # img_rgb = cv2.cvtColor(self.modified_img, cv2.COLOR_BGR2RGB)
        r,g,b = cv2.split(self.modified_img)
        # scaler = self.scaler2.get()
        R = self.lowpass_filter(r)
        G = self.lowpass_filter(g)
        B = self.lowpass_filter(b)
        self.modified_img = cv2.merge((R, G, B)) 
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.array(self.modified_img, dtype=np.uint8)))
        
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def lowpass_filter(self, img):
        F = np.fft.fft2(img)
        n= self.scaler1.get()
        D0= self.scaler2.get()

        F=np.fft.fftshift(F)
        M, N = img.shape
        u = np.arange(0, M)-M/2
        v = np.arange(0,N)-N/2
        [V, U] = np.meshgrid(v, u)
        D = np.sqrt(np.power(U,2) + np.power(V,2))
        H = 1/np.power(1+ (D/D0), (2*n))
        G = H*F
        G = np.fft.ifftshift(G)
        imgOut = np.real(np.fft.ifft2(G))
        return imgOut
        # self.modified_img = imgOut

        # self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        # self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_butterworth_color(self):
        img_rgb = cv2.cvtColor(self.modified_img, cv2.COLOR_BGR2RGB)

        # self.modified_img = cv2.imread(self.image_path, 0)
        # self.modified_img = cv2.resize(self.modified_img, (int(self.img_width), int(self.img_height)), interpolation= cv2.INTER_LINEAR)
        
        r,g,b = cv2.split(img_rgb)
        # scaler = self.scaler2.get()
        R = self.highpass_butterworth(r)
        G = self.highpass_butterworth(g)
        B = self.highpass_butterworth(b)
        self.modified_img = cv2.merge((R, G, B)) 
        # self.modified_img = self.highpass_butterworth(self.modified_img)
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.array(self.modified_img, dtype=np.uint8)))
        
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_butterworth(self, img):
        # F = np.fft.fft2(img)
        # M, N = img.shape
        # n = self.scaler1.get()
        # D0 = self.scaler2.get()
        # u = np.arange(0, M, 1)
        # v = np.arange(0,N,1)
        # idx = (u > M/2)
        # u[idx] = u[idx] - M
        # idy = (v > N/2)
        # v[idy] = v[idy] - N
        # [V, U] = np.meshgrid(v, u)
        # D = np.sqrt(np.power(U, 2) + np.power(V, 2))
        # H = 1/np.power(1 + (D0/(D+1e-10)), 2*n)
        # G = H * F
        # imgOut = np.real(np.fft.ifft2(G))
        # return imgOut

        F = np.fft.fft2(img)
        F_shift = np.fft.fftshift(F)
        D0 = self.scaler2.get()

        M, N = img.shape
        H = np.zeros((M, N), dtype=np.float32)
        n = 1
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H[u, v] = 1 / (1 + (D0/D)**n)

        G_shift = F_shift * H

        G = np.fft.ifftshift(G_shift)
        g = np.abs(np.fft.ifft2(G))
        g = np.clip(g, 0, 255)

        return g

        # self.modified_img = imgOut

        # self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        # self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_ideal_color(self):
        img_rgb = cv2.cvtColor(self.modified_img, cv2.COLOR_BGR2RGB)
        r,g,b = cv2.split(img_rgb)
        # scaler = self.scaler2.get()
        R = self.highpass_ideal(r)
        G = self.highpass_ideal(g)
        B = self.highpass_ideal(b)
        self.modified_img = cv2.merge((R, G, B)) 
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.array(self.modified_img, dtype=np.uint8)))
        
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_ideal(self, img):
        D0 = self.scaler2.get()
        F = np.fft.fft2(img)
        F_shift = np.fft.fftshift(F)

        M, N = img.shape

        H = np.zeros((M, N), dtype=np.float32)

        for u in range(M):
            for v in range(N):
                D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
                if D <= D0:
                    H[u, v] = 0
                else:
                    H[u, v] = 1

        G_shift = F_shift * H

        G = np.fft.ifftshift(G_shift)
        g = np.abs(np.fft.ifft2(G))
        g = np.clip(g, 0, 255)

        return g
    
        F = np.fft.fft2(img)
        M, N = img.shape
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
        H = np.double(D > D0)
        G = H * F
        imgOut = np.real(np.fft.ifft2(G))
        return imgOut
        # self.modified_img = imgOut

        # self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        # self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def highpass_gaussian(self):
        pass


    def erode_img(self):
        # img_rgb = cv2.cvtColor(self.modified_img, cv2.COLOR_BGR2RGB)
        # r,g,b = cv2.split(img_rgb)
        kernel = np.ones((self.scaler3.get(), self.scaler3.get()), np.uint8)
        # R = cv2.erode(r, kernel=kernel)
        # G = cv2.erode(g, kernel=kernel)
        # B = cv2.erode(b, kernel=kernel)
        # self.modified_img = cv2.merge((R, G, B))
        self.modified_img = cv2.erode(self.modified_img, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    # def erode_img(self, img, kernel):
    #     return cv2.erode(img, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        

    def dilate_img(self):
        kernel = np.ones((self.scaler3.get(), self.scaler3.get()), np.uint8)
        self.modified_img = cv2.dilate(self.modified_img, kernel=kernel)
        self.modified_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.modified_img))
        self.canvas_modified_img.create_image(0, 0, image=self.modified_photo, anchor=tkinter.NW)

    def opening_img(self):
        self.dilate_img()
        self.erode_img()

    def closing_img(self):
        self.erode_img()
        self.dilate_img()
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")