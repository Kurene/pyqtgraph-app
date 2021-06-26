# -*- coding: utf-8 -*-
import sys
import time
import threading
import numpy as np
from numba import jit
import librosa
"""
import os
import PySide6
from PySide6 import QtGui, QtCore
dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
"""
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

from rasp_audio_stream import AudioInputStream


class PQGMelSpectrogram():
    def __init__(self,
                 sr,
                 shape,
                 n_mels=128,
                 n_frames=150,
                 fps=60, 
                 size=(500,500), 
                 title="",
        ):
        # メルスペクトログラム算出用パラメタ
        self.n_frames  = n_frames
        self.n_ch      = shape[0]
        self.n_chunk   = shape[1]
        self.n_freqs   = self.n_chunk // 2 + 1
        self.n_mels      = n_mels
        self.sig       = np.zeros(shape)
        self.x         = np.zeros(self.n_chunk)
        self.specs     = np.zeros((self.n_freqs))
        self.melspecs  = np.zeros((self.n_frames, self.n_mels))
        self.window    = np.hamming(self.n_chunk)
        self.fft       = np.fft.rfft
        self.melfreqs  = librosa.mel_frequencies(n_mels=self.n_mels)
        self.melfb     = librosa.filters.mel(sr, self.n_chunk, n_mels=self.n_mels)
        self.fps       = fps
        self.iter      = 0
        
        #====================================================
        ## PyQtGraph の初期設定
        app = QtGui.QApplication([]) 
        win = pg.GraphicsLayoutWidget()
        win.resize(size[0], size[1])
        win.show()
        
        ## ImageItem の設定
        imageitem = pg.ImageItem(border="k")
        cmap = pg.colormap.getFromMatplotlib("jet")
        bar = pg.ColorBarItem( cmap=cmap )
        bar.setImageItem(imageitem) 
        
        ## ViewBox の設定
        viewbox = win.addViewBox()
        viewbox.setAspectLocked(lock=True)
        viewbox.addItem(imageitem)
  
        ## 軸 (AxisItem) の設定
        axis_left = pg.AxisItem(orientation="left")
        n_ygrid = 6
        yticks = {}
        for k in range(n_ygrid):
            index = k*(self.n_mels//n_ygrid)
            yticks[index] = int(self.melfreqs[index])
        axis_left.setTicks([yticks.items()])
        
        ## PlotItemの設定
        plotitem = pg.PlotItem(viewBox=viewbox, axisItems={"left":axis_left})
        # グラフの範囲
        plotitem.setLimits(
            minXRange=0, maxXRange=self.n_frames, 
            minYRange=0, maxYRange=self.n_mels)
        # アスペクト比固定
        plotitem.setAspectLocked(lock=True)
        # マウス操作無効
        plotitem.setMouseEnabled(x=False, y=False)
        # ラベルのセット
        plotitem.setLabels(bottom="Time-frame", 
                           left="Frequency")
        win.addItem(plotitem)
        
        self.app       = app
        self.win       = win
        self.viewbox   = viewbox
        self.plotitem  = plotitem
        self.imageitem = imageitem
        
        pg.setConfigOptions(antialias=True)
        #pg.setConfigOption('useNumba', True)
        
    def run_app(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(1/self.fps * 1000)
        
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):
        if self.iter > 0:
            self.viewbox.disableAutoRange()
        
        # 最新をスペクトログラム格納するインデックス
        idx = self.iter % self.n_frames
        # モノラル信号算出
        self.x[:] = 0.5 * (self.sig[0] + self.sig[1])
        # FFT => パワー算出
        self.x[:] = self.x[:] * self.window
        self.specs[:] = np.abs(self.fft(self.x))**2
        # メルスペクトログラム算出
        self.melspecs[idx, :] = np.dot(self.melfb, self.specs)
        
        # 描画
        pos = idx + 1 if idx < self.n_frames else 0
        self.imageitem.setImage(
                librosa.power_to_db(
                    np.r_[self.melspecs[pos:self.n_frames], 
                    self.melspecs[0:pos]]
                , ref=np.max)
            )
        self.iter += 1
        
        
    def callback_sigproc(self, sig):
        self.sig[:] = sig