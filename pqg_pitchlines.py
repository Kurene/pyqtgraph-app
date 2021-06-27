# -*- coding: utf-8 -*-
import sys
import time
import threading
import numpy as np
from numba import jit
import librosa

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg


# 散布図プロット用クラス
class PQGPitchLines():
    def __init__(self,
                 sr,
                 sig_shape,
                 n_frames=150,
                 fps=60, 
                 size=(500,500), 
                 title=""
        ):
        self.n_ch, self.n_chunk = sig_shape
        self.n_frames   = n_frames
        self.n_chroma   = 12
        self.n_freqs    = self.n_chunk // 2 + 1
        self.sig        = np.zeros(sig_shape)
        self.sig_mono   = np.zeros(self.n_chunk)
        self.specs      = np.zeros(self.n_freqs)
        self.chroma_pre = np.zeros(self.n_chroma)
        self.chroma     = np.zeros(self.n_chroma)
        self.window     = np.hamming(self.n_chunk)
        self.fft        = np.fft.rfft        
        self.chromafb   = librosa.filters.chroma(sr, self.n_chunk, tuning=0.0, n_chroma=self.n_chroma)
        self.chromafb **= 2
        self.x = np.arange(0, self.n_frames)
        self.y = np.zeros((self.n_frames, self.n_chroma))
           
        # PyQtGraph 散布図の初期設定
        self.app = QtGui.QApplication([]) 
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(size[0], size[1])
        self.win.show()
       
        self.plotitem = self.win.addPlot(title=title)
        self.plotitem.setXRange(0, self.n_frames)
        self.plotitem.setYRange(0, self.n_chroma)
        self.plotitem.showGrid(x=False, y=False)
        
        self.plots = []
        for k in range(self.n_chroma):
            self.plots.append(
                self.plotitem.plot(pen=pg.mkPen((k, self.n_chroma), width=3)) 
            )
        
        self.fps = fps
        self.iter = 0
        
        pg.setConfigOptions(antialias=True)
        
    def update(self):
        idx = self.iter % self.n_frames

        self.sig_mono[:] = 0.5 * (self.sig[0] + self.sig[1])
        pw = np.sqrt(np.mean(self.sig_mono**2))
        self.sig_mono[:] = self.sig_mono[:] * self.window
        self.specs[:] = np.abs(self.fft(self.sig_mono))**2
        self.chroma[:] = np.dot(self.chromafb, self.specs)
        self.chroma[:] = self.chroma / (np.max(self.chroma)+1e-16)
        self.chroma[:] = 0.3*self.chroma+0.7*self.chroma_pre
        self.y[idx]    = self.chroma[:]
        
        #print(self.y[idx])
        pos = idx + 1 if idx < self.n_frames else 0
        for k in range(self.n_chroma):
            alpha = self.y[idx,k] * 0.9
            alpha = alpha if pw > 1e-3 else 0.0
            self.plots[k].setAlpha(alpha, False)
            self.y[idx,k] += k
            self.plots[k].setData(
                self.x, 
                np.r_[self.y[pos:self.n_frames,k],self.y[0:pos,k]]
                )
            
        self.chroma_pre[:] = self.chroma
        self.iter += 1


    def run_app(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(1/self.fps * 1000)
        
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    
    def callback_sigproc(self, sig):
        self.sig[:] = sig
