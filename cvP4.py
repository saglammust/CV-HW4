#   Mustafa Sağlam  #
#   150140129       #
#   Computer Vision #HW4
#   22.12.2018      #

from PyQt5.uic.properties import QtGui
from PyQt5.QtWidgets import (QAction, QWidget, QLabel, QMainWindow, QPushButton, QApplication, QMenu)
from PyQt5.QtWidgets import (QVBoxLayout, QGroupBox, QFileDialog, QMessageBox, QSizePolicy, QWidget)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import harris_corner as hc
import k2means as k2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

class window(QMainWindow):
    imgHCgray = None
    imgMRgray = None
    imgHCcolor = None
    imgMRcolor = None

    def __init__(self, parent=None):
        super(window, self).__init__(parent)
        self.setGeometry(55,25,960,640)
        self.setWindowTitle('Harris-Corner Detector and K-Means Application')
        
        actionQuit = QAction('&Exit', self)
        actionQuit.setShortcut('Ctrl+Q')
        actionQuit.triggered.connect(self.close_app)

        actionLoadImgHC = QAction("&Load Image for Harris Corner Detect",self)
        actionLoadImgHC.setShortcut('Ctrl+H')
        actionLoadImgHC.triggered.connect(self.load_imgHC)

        actionLoadImgMR = QAction("&Load Image for K-means Application", self)
        actionLoadImgMR.setShortcut('Ctrl+K')
        actionLoadImgMR.triggered.connect(self.load_imgMR)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(actionLoadImgHC)
        fileMenu.addAction(actionLoadImgMR)
        fileMenu.addAction(actionQuit)

        actionRunHC = QAction('Run Harris Corner Detection',self)
        actionRunHC.triggered.connect(self.run_main_harris)
        actionRunKmeans = QAction('Run K-means Algorithm', self)
        actionRunKmeans.triggered.connect(self.run_main_k2means)

        self.toolBar = self.addToolBar('Toolbar')
        self.toolBar.addAction(actionRunHC)
        self.toolBar.addAction(actionRunKmeans)

        self.initUI()

    def initUI(self):
        self.gb_input = QGroupBox('Harris Corner Detection',self)
        self.gb_input.setStyleSheet('border: 1px;'
                 'QGroupBox:title {'
                 'subcontrol-origin: margin;'
                 'subcontrol-position: top left;'
                 'padding-left: 3px;'
                 'padding-right: 3px;'
                 'margin-left: 10px }')
        self.gb_input.resize(275,550)
        self.gb_input.move(40,50)

        self.gb_target = QGroupBox('K-Means Application(k=2)',self)
        self.gb_target.setStyleSheet('border: 1px;'
                 'QGroupBox:title {'
                 'subcontrol-origin: margin;'
                 'subcontrol-position: top left;'
                 'padding-left: 3px;'
                 'padding-right: 3px;'
                 'margin-left: 10px }')
        self.gb_target.resize(550,550)
        self.gb_target.move(350,50)
        
        self.show()

    def close_app(self):
        sys.exit()

    def load_imgHC(self):
        imgHC = QLabel(self)
        path = QFileDialog.getOpenFileName(self, 'Open file','',"Image files (*.jpg *.png)")
        
        try:
            self.imgHCgray = cv2.imread(path[0], 0)
            self.imgHCcolor = cv2.imread(path[0], 1)
            h,w = self.imgHCgray.shape
        except AttributeError:
            try:
                self.imgHCgray = cv2.imread("blocks.jpg",0)
                self.imgHCcolor = cv2.imread("blocks.jpg",1)
                h,w = self.imgHCgray.shape
            except AttributeError:
                print("OpenCV cold not open the images. The application will be closed.")
                sys.exit(1)

        image = QImage(self.imgHCcolor, w, h, w*3, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(image)
        pix_scaled = pix.scaled(255,255, Qt.KeepAspectRatio)
        imgHC.setPixmap(pix_scaled)
        imgHC.resize(pix_scaled.width(),pix_scaled.height())
        imgHC.move(50,65)

        imgHC.show()

    def load_imgMR(self):
        imgMR = QLabel(self)
        path = QFileDialog.getOpenFileName(self, 'Open file','',"Image files (*.jpg *.png)")
        self.imgMRgray = cv2.imread(path[0],0)
        self.imgMRcolor = cv2.imread(path[0],1)
        
        try:
            self.imgMRgray = cv2.imread(path[0], 0)
            self.imgMRcolor = cv2.imread(path[0], 1)
            h,w = self.imgMRgray.shape
        except AttributeError:
            try:
                self.imgMRgray = cv2.imread("mr.jpg",0)
                self.imgMRcolor = cv2.imread("mr.jpg",1)
                h,w = self.imgMRgray.shape
            except AttributeError:
                print("OpenCV could not open the images. The application will be closed.")
                sys.exit(1)

        image = QImage(self.imgMRcolor, w, h, w*3, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(image)
        pix_scaled = pix.scaled(255,255, Qt.KeepAspectRatio)
        imgMR.setPixmap(pix_scaled)
        imgMR.resize(pix_scaled.width(),pix_scaled.height())
        imgMR.move(360,65)

        imgMR.show()

    def run_main_harris(self):
        if (type(self.imgHCgray) == type(None)):
            mistake = QMessageBox.warning(self, 'Crucial Mistake', 'You have not loaded the image!\n'
                'Please load the image and try again!', QMessageBox.Ok)
            if mistake == QMessageBox.Ok:
                return

        imgHCfiltered = hc.gaussian_filter(self.imgHCgray)
        Ix, Iy = hc.gradientXY(self.imgHCgray)
        points = hc.harris_detector(Ix, Iy, imgHCfiltered)
        h,w = self.imgHCgray.shape
        pcolor = (25, 55, 245)#BGR
        rect = (0,0,w,h)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert(p)
        for p in points:
            cv2.circle(self.imgHCcolor, p, 1, pcolor, -1, cv2.LINE_AA, 0)
        
        imgHC = QLabel(self)
        image = QImage(self.imgHCcolor, w, h, w*3, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(image)
        pix_scaled = pix.scaled(255,255, Qt.KeepAspectRatio)
        imgHC.setPixmap(pix_scaled)
        imgHC.resize(pix_scaled.width(),pix_scaled.height())
        imgHC.move(50,335)

        imgHC.show()

    def run_main_k2means(self):
        if (type(self.imgMRgray) == type(None)):
            mistake = QMessageBox.warning(self, 'Crucial Mistake', 'You have not loaded the image!\n'
                'Please load the image and try again!', QMessageBox.Ok)
            if mistake == QMessageBox.Ok:
                return

        imgMRmask1 = k2.binary_mask(self.imgMRgray)
        imgMRbrain = k2.morph_mask(imgMRmask1)
        imgMRk2run = k2.k2means(self.imgMRgray,imgMRbrain)

        points = k2.prewitt_edge(imgMRk2run)
        h,w = self.imgMRgray.shape
        rect = (0,0,w,h)
        pcolor = (255,55,55)#BGR
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert(p)
        for p in points:
            cv2.circle(self.imgMRcolor, p, 1, pcolor, -1, cv2.LINE_AA, 0)

        imgMask = QLabel(self)
        imageMask = QImage(imgMRmask1, w, h, w, QImage.Format_Grayscale8)
        pixMask = QPixmap(imageMask)
        pixMask_scaled = pixMask.scaled(255,255, Qt.KeepAspectRatio)
        imgMask.setPixmap(pixMask_scaled)
        imgMask.resize(pixMask_scaled.width(),pixMask_scaled.height())
        imgMask.move(635,65)

        imgMask.show()

        imgK2 = QLabel(self)
        imageK2 = QImage(imgMRk2run, w, h, w, QImage.Format_Grayscale8)
        pixK2 = QPixmap(imageK2)
        pixK2_scaled = pixK2.scaled(255,255, Qt.KeepAspectRatio)
        imgK2.setPixmap(pixK2_scaled)
        imgK2.resize(pixK2_scaled.width(),pixK2_scaled.height())
        imgK2.move(360,335)

        imgK2.show()

        imgMR = QLabel(self)
        imageMR = QImage(self.imgMRcolor, w, h, w*3, QImage.Format_RGB888).rgbSwapped()
        pixMR = QPixmap(imageMR)
        pixMR_scaled = pixMR.scaled(255,255, Qt.KeepAspectRatio)
        imgMR.setPixmap(pixMR_scaled)
        imgMR.resize(pixMR_scaled.width(),pixMR_scaled.height())
        imgMR.move(635,335)

        imgMR.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = window()
    sys.exit(app.exec_())