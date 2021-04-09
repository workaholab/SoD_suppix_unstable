import math
from skimage import io, color
import numpy as np
from tqdm import trange

# RGBD_slic===============================================================
class RGBD_Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0, d=0, dgrad=0):
        self.update(h, w, l, a, b, d, dgrad)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b, d, dgrad):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
        self.d = d
        self.dgrad = dgrad

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()
#  ===============================================================       
class RGBD_SLICProcessor(object):
    def __init__(self, filename, depth_filename,  K=4096, M=40):
        self.K = K
        self.M = M

        self.img_data, self.dep_data = self.open_image(filename, depth_filename)
        self.image_height = self.img_data.shape[0]
        self.image_width = self.img_data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf) 
        
        self.iterate_10times() 
        
        return self.clusters

    @staticmethod
    def open_image(path, path_dep):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        
        depth = io.imread(path_dep) ##
        dep_amp = color.rgb2gray(depth) ##
        
        return lab_arr, dep_amp

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return RGBD_Cluster(h, w,
                       self.img_data[h][w][0],
                       self.img_data[h][w][1],
                       self.img_data[h][w][2],
                       self.dep_data[h][w])

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.img_data[h + 1][w + 1][0] - self.img_data[h][w][0] + \
                   self.img_data[h + 1][w + 1][1] - self.img_data[h][w][1] + \
                   self.img_data[h + 1][w + 1][2] - self.img_data[h][w][2]
        return gradient
    
    def get_gradient_ddgrad(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        d_gradient = float(self.dep_data[h + 1][w + 1]) - float(self.dep_data[h][w])        
        
        return d_gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            # d_gradient= self.get_gradient_ddgrad(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    # new_dgradient= self.get_gradient_ddgrad(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, 
                                        self.img_data[_h][_w][0], 
                                        self.img_data[_h][_w][1], 
                                        self.img_data[_h][_w][2], 
                                        self.dep_data[_h][_w], 
                                        self.get_gradient_ddgrad(_h, _w) )
                        cluster_gradient = new_gradient 

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.img_data[h][w]
                    Damp = self.dep_data[h][w]
                    d_gradient = self.get_gradient_ddgrad(h, w)
                    
                    #Distance calculation
                    # lab, spatial
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    
                    # depth distance
                    Dddg=math.sqrt(
                        math.pow( int(Damp) - int(cluster.d), 2) +
                        math.pow( float(d_gradient) - float(cluster.dgrad), 2))
                        
                    D = math.sqrt( math.pow(Dc , 2) + #/ self.M
                                   math.pow(Ds / self.S, 2)*1024 + 
                                   math.pow(Dddg, 2) )
                   
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.img_data[_h][_w][0], self.img_data[_h][_w][1], self.img_data[_h][_w][2], self.dep_data[_h][_w], self.get_gradient_ddgrad(_h, _w) )

    def save_current_image(self, name):
        image_arr = np.copy(self.img_data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    
    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            '''
            name = 'outRGBD_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)
            '''
#  ===============================================================
class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1
        
        # self.iterate_10times() 
        
        # return self.clusters

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()

#  ===============================================================
class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K=4096, M=40):
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)
        
        self.iterate_10times() 
        
        return self.clusters

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        # print(self.clusters)
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
    '''
    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)
    '''
    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            
            #name = 'out_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            #self.save_current_image(name)

