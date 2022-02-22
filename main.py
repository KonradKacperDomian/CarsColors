##
# @mainpage Skrypt sluzy do okreslenia koloru samochodu
#
# @selection description_main Description
# Skrypt Python do sekrekacji samochodów na podstawie ich koloru
#
# @author Konrad Domian
##
# @file main.py
#
# @brief Skrypt używa biblioteki openCV do wszelkich operaji
#
# @selection libraries_main Libraries/Modules
# -cv2
# -numpy
# -sklearn
###
#imports
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

def FindDominantColors(cluster, centroids):
    """
    Funkcja służy do znalezienia dominujacych kolorów, do tego używa alhgorytmu Kmean
    :param cluster: klaster
    :param centroids: centralny klaster
    :return: DomiantColors[-2]: jest to drugi co do wartości procentowej udziału kolor w obrazie, uzyty został drugi
    ponieważ została zastosowana metoda do wycinania obiektu pierwszo planowego
    """
    DominantColors = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    for (percent, color) in colors:
        DominantColors.append(color)
    return DominantColors[-2] #Background==[0,0,0] => DominatColor==Black

def ReadPhotoAndRemoveBackground(filename):
    """
    Funkcja służy do zastepowania pikseli tła czarnymi pikselami
    :param filename: nazwa pliku z obrazem
    :return: img: cv2 img
    """
    img = cv.imread(filename)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w, c = img.shape
    rect = (0, 0, w - 50, h - 50)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img

if __name__ == "__main__":
    img = ReadPhotoAndRemoveBackground('black17.jpeg')
    reshape = img.reshape((img.shape[0] * img.shape[1], 3))
    cluster = KMeans(n_clusters=5).fit(reshape)
    DominantColor = FindDominantColors(cluster, cluster.cluster_centers_)
    print("Color (BGR):" + str(DominantColor))

