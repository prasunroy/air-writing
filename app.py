# -*- coding: utf-8 -*-
"""
Air-Writing.
Created on Wed May  9 22:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/air-writing

"""


# imports
import sys
import webbrowser

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QWidget
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QDesktopWidget, QLabel, QPushButton

from camera import VideoStream
from pipeline import Pipeline


# MainGUI class
class MainGUI(QWidget):
    
    # ~~~~~~~~ constructor ~~~~~~~~
    def __init__(self):
        super().__init__()
        self.init_pipeline()
        self.init_UI()
        
        return
    
    # ~~~~~~~~ initialize pipeline ~~~~~~~~
    def init_pipeline(self):
        self.pipeline = Pipeline()
        self.engine = 'EN'
        
        return
    
    # ~~~~~~~~ initialize ui ~~~~~~~~
    def init_UI(self):
        # set properties
        self.setGeometry(0, 0, 0, 0)
        self.setStyleSheet('QWidget {background-color: #ffffff;}')
        self.setWindowIcon(QIcon('assets/logo.png'))
        self.setWindowTitle('Air-Writing')
        
        # create widgets
        # -- connect camera button --
        self.btn_conn = QPushButton('Connect Camera')
        self.btn_conn.setMinimumSize(500, 40)
        self.btn_conn_style_0 = 'QPushButton {background-color: #00a86c; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn_style_1 = 'QPushButton {background-color: #ff6464; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 16px;}'
        self.btn_conn.setStyleSheet(self.btn_conn_style_0)
        
        # -- camera feed --
        self.cam_feed = QLabel()
        self.cam_feed.setMinimumSize(640, 480)
        self.cam_feed.setAlignment(Qt.AlignCenter)
        self.cam_feed.setFrameStyle(QFrame.StyledPanel)
        self.cam_feed.setStyleSheet('QLabel {background-color: #000000;}')
        
        # ---- recognition engine selection buttons ----
        self.btn_engine_style_0 = 'QPushButton {background-color: #646464; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 14px;}'
        self.btn_engine_style_1 = 'QPushButton {background-color: #6464ff; border: none; color: #ffffff; font-family: ubuntu, arial; font-size: 14px;}'
        self.btn_en = QPushButton('English Numerals')
        self.btn_en.setMinimumSize(150, 30)
        self.btn_en.setStyleSheet(self.btn_engine_style_1)
        self.btn_bn = QPushButton('Bengali Numerals')
        self.btn_bn.setMinimumSize(150, 30)
        self.btn_bn.setStyleSheet(self.btn_engine_style_0)
        self.btn_dv = QPushButton('Devanagari Numerals')
        self.btn_dv.setMinimumSize(150, 30)
        self.btn_dv.setStyleSheet(self.btn_engine_style_0)
        
        # -- repository link button --
        self.btn_repo = QPushButton()
        self.btn_repo.setFixedSize(20, 20)
        self.btn_repo.setStyleSheet('QPushButton {background-color: none; border: none;}')
        self.btn_repo.setIcon(QIcon('assets/button_repo.png'))
        self.btn_repo.setIconSize(QSize(20, 20))
        self.btn_repo.setToolTip('Fork me on GitHub')
        
        # -- copyright --
        self.copyright = QLabel('\u00A9 2018 Indian Staistical Institute')
        self.copyright.setFixedHeight(20)
        self.copyright.setAlignment(Qt.AlignCenter)
        self.copyright.setStyleSheet('QLabel {background-color: #ffffff; font-family: ubuntu, arial; font-size: 14px;}')
        
        # create layouts
        h_box1 = QHBoxLayout()
        h_box1.addWidget(self.btn_conn)
        
        h_box2 = QHBoxLayout()
        h_box2.addWidget(self.btn_en)
        h_box2.addWidget(self.btn_bn)
        h_box2.addWidget(self.btn_dv)
        
        h_box3 = QHBoxLayout()
        h_box3.addWidget(self.btn_repo)
        h_box3.addWidget(self.copyright)
        
        v_box1 = QVBoxLayout()
        v_box1.addLayout(h_box1)
        v_box1.addLayout(h_box2)
        v_box1.addStretch()
        v_box1.addLayout(h_box3)
        
        v_box2 = QVBoxLayout()
        v_box2.addWidget(self.cam_feed)
        
        g_box0 = QGridLayout()
        g_box0.addLayout(v_box1, 0, 0, -1, 2)
        g_box0.addLayout(v_box2, 0, 2, -1, 4)
        
        self.setLayout(g_box0)
        
        # set slots for signals
        self.flg_conn = False
        self.btn_conn.clicked.connect(self.connect)
        self.btn_en.clicked.connect(lambda: self.setRecognitionEngine('EN'))
        self.btn_bn.clicked.connect(lambda: self.setRecognitionEngine('BN'))
        self.btn_dv.clicked.connect(lambda: self.setRecognitionEngine('DV'))
        self.btn_repo.clicked.connect(self.openRepository)
        
        return
    
    # ~~~~~~~~ window centering ~~~~~~~~
    def moveWindowToCenter(self):
        window_rect = self.frameGeometry()
        screen_cent = QDesktopWidget().availableGeometry().center()
        window_rect.moveCenter(screen_cent)
        self.move(window_rect.topLeft())
        
        return
    
    # ~~~~~~~~ connect camera ~~~~~~~~
    def connect(self):
        self.flg_conn = not self.flg_conn
        if self.flg_conn:
            self.btn_conn.setStyleSheet(self.btn_conn_style_1)
            self.btn_conn.setText('Disconnect Camera')
            self.video = VideoStream()
            self.timer = QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(50)
        else:
            self.btn_conn.setStyleSheet(self.btn_conn_style_0)
            self.btn_conn.setText('Connect Camera')
            self.cam_feed.clear()
            self.timer.stop()
            self.video.clear()
        
        return
    
    # ~~~~~~~~ update ~~~~~~~~
    def update(self):
        frame = self.video.getFrame(flip=1)
        if not frame is None:
            prediction, predprobas, mask, frame = self.pipeline.run_inference(frame, self.engine)
            print(prediction, predprobas)
            frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.cam_feed.setPixmap(QPixmap.fromImage(frame))
        else:
            self.cam_feed.clear()
        
        return
    
    # ~~~~~~~~ set recognition engine ~~~~~~~~
    def setRecognitionEngine(self, engine='EN'):
        if engine.upper() == 'EN':
            self.engine = 'EN'
            self.btn_en.setStyleSheet(self.btn_engine_style_1)
            self.btn_bn.setStyleSheet(self.btn_engine_style_0)
            self.btn_dv.setStyleSheet(self.btn_engine_style_0)
        elif engine.upper() == 'BN':
            self.engine = 'BN'
            self.btn_en.setStyleSheet(self.btn_engine_style_0)
            self.btn_bn.setStyleSheet(self.btn_engine_style_1)
            self.btn_dv.setStyleSheet(self.btn_engine_style_0)
        elif engine.upper() == 'DV':
            self.engine = 'DV'
            self.btn_en.setStyleSheet(self.btn_engine_style_0)
            self.btn_bn.setStyleSheet(self.btn_engine_style_0)
            self.btn_dv.setStyleSheet(self.btn_engine_style_1)
        
        return
    
    # ~~~~~~~~ open repository ~~~~~~~~
    def openRepository(self):
        webbrowser.open('https://github.com/prasunroy/air-writing')
        
        return
    
    # ~~~~~~~~ close event ~~~~~~~~
    def closeEvent(self, event):
        if self.flg_conn:
            self.connect()
        
        return


# main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = MainGUI()
    gui.show()
    gui.setFixedSize(gui.size())
    gui.moveWindowToCenter()
    sys.exit(app.exec_())
