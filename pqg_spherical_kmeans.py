# -*- coding: utf-8 -*-
import sys
import time
import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


# 散布図プロット用クラス
class KMeans2D():
    def __init__(self,
                 n_samples,
                 n_labels,
                 spherical_mode=True,
                 max_iter=100,
                 fps=1, 
                 xrange=[-1.5, 1.5], 
                 yrange=[-1.5, 1.5],
                 size=(500,500), 
                 title="",
        ):
        self.n_samples = n_samples
        self.n_labels = n_labels
        self.fps = fps
        
        # PyQtGraph
        self.app = QtGui.QApplication([]) 
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(size[0], size[1])
        self.win.show()
        
        self.plotitem = self.win.addPlot(title=title)
        self.plotitem.setXRange(xrange[0], xrange[1])
        self.plotitem.setYRange(yrange[0], yrange[1])
        self.plotitem.showGrid(x = True, y = True, alpha = 0.3)
        
        self.plot_data = []
        for k in range(self.n_labels):
            self.plot_data.append(
                self.plotitem.plot(
                    pen=None, 
                    symbol="o", 
                    symbolPen='b', 
                    symbolSize=10, 
                    symbolBrush=pg.mkBrush(pg.intColor(k, hues=self.n_labels))
                ) 
            )
        self.plot_centroids = self.plotitem.plot(
                    pen=None, 
                    symbol="x", 
                    symbolPen='b', 
                    symbolSize=10, 
                    symbolBrush="c"
        ) 
        pg.setConfigOptions(antialias=True)

        if spherical_mode:
            self.calc_init_centroids = calc_init_centroids_cos
            self.distance            = distance_cos
            self.calc_centroid       = calc_centroid_cos
        else:
            self.calc_init_centroids = calc_init_centroids_euc
            self.distance            = distance_euc
            self.calc_centroid       = calc_centroid_euc
            
            
    def run(self, xy):
        self.iter = 0
        self.xy = xy.copy()
        self.labels = np.zeros(self.n_samples)
        self.centroids = np.zeros((self.n_labels, 2))
        self.calc_init_centroids(self.centroids, self.n_labels)
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(int(1/self.fps * 1000))
        
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):
        if self.iter % 2 == 0:
            for k in range(self.n_samples):
                dist = 1e15
                for m in range(self.n_labels):
                    tmp_dist = self.distance(self.xy[k], self.centroids[m])
                    if tmp_dist < dist:
                        dist = tmp_dist
                        self.labels[k] = m
        else:
            for m in range(self.n_labels):
                tmp_xy = self.xy[self.labels==m]
                length = tmp_xy.shape[0]
                self.calc_centroid(tmp_xy, self.centroids, m)
            
        # Draw
        for m in range(self.n_labels):
            tmp_xy = self.xy[self.labels==m]
            if tmp_xy.shape[0] > 0:
                self.plot_data[m].setData(tmp_xy[:,0], tmp_xy[:,1])
                self.plot_data[m].setAlpha(0.7, False)
        self.plot_centroids.setData(
            self.centroids[:,0], self.centroids[:,1])
            
        self.iter += 1

def calc_init_centroids_cos(centroids, n_labels):
    for k in range(n_labels):
        theta = np.random.random(1)*2*np.pi
        centroids[k,0] = np.cos(theta)
        centroids[k,1] = np.sin(theta)
        
def calc_init_centroids_euc(centroids, n_labels):
    for k in range(n_labels):
        centroids[k] = np.random.random(2)*2-1
        
def distance_cos(x, v):
    return 1.0 - np.dot(x, v)

def distance_euc(x, v):
    return np.sum((x-v)**2)

def calc_centroid_cos(x, v, m):
    v[m] = np.sum(x, axis=0)
    v[m] /= np.linalg.norm(v[m]) + 1e-15    

def calc_centroid_euc(x, v, m):
    v[m] = np.mean(x, axis=0)


if __name__ == "__main__":
    n_samples = int(sys.argv[1])
    n_labels = int(sys.argv[2])
    mode = int(sys.argv[3])
    
    x = np.random.normal(0.0, 0.5, (n_samples,2))
    spherical_mode = True if mode == 0 else False
    kmeans2d = KMeans2D(x.shape[0], n_labels, spherical_mode=spherical_mode)
    kmeans2d.run(x)