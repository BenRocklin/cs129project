import numpy as np
from collections import defaultdict
import csv
import os
import math
from skimage import io, feature
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from scipy import ndimage as ndi

LEFT_EYE = 0
RIGHT_EYE = 1
MOUTH = 2
EAR_ONE = 3
EAR_TWO = 4
EAR_THREE = 5
EAR_FOUR = 6
EAR_FIVE = 7
TARGET_ROWS = 150
TARGET_COLS = 150

def imageToFeatures(imageMatrix, file):
    grayscale = rgb2gray(imageMatrix)
    grayFace = getIsolatedRotatedFace(file, grayscale)

    gausImage = ndi.gaussian_filter(grayFace, 1.5)
    newGrayEdges = feature.canny(gausImage, sigma=0.01)

    adjusted = getResizedCannyFace(newGrayEdges)

    #resizedImage = resize(grayscale, (375, 500))

    #ks = 5
    #sig = 1.4
    #h = 0.07
    #l = 0.04
    #grayEdges, GxGray, GyGray, smoothed, nms, strong_edges, weak_edges = canny(resizedImage, kernel_size=ks, sigma=sig, high=h, low=l)
    
    #gausImage = ndi.gaussian_filter(resizedImage, 1.5)
    #grayEdges = feature.canny(gausImage, sigma=0.01)

    return adjusted.flatten()


def findAngle(leftEye, rightEye):
    myradians = math.atan2(rightEye[1]-leftEye[1], rightEye[0]-leftEye[0])
    mydegrees = math.degrees(myradians)
    return mydegrees


def newRotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)


def getCoordInfo(fullPath):
    allData = list()
    with open(fullPath + ".cat") as f:
        for line in f:
            allData = line.split()

    coordDict = dict()
    leftEye = (int(allData[1]), int(allData[2]))
    rightEye = (int(allData[3]), int(allData[4]))
    mouthC = (int(allData[5]), int(allData[6]))
    earOne = (int(allData[7]), int(allData[8]))
    earTwo = (int(allData[9]), int(allData[10]))
    earThree = (int(allData[11]), int(allData[12]))
    earFour = (int(allData[13]), int(allData[14]))
    earFive = (int(allData[15]), int(allData[16]))

    coordDict[LEFT_EYE] = leftEye
    coordDict[RIGHT_EYE] = rightEye
    coordDict[MOUTH] = mouthC
    coordDict[EAR_ONE] = earOne
    coordDict[EAR_TWO] = earTwo
    coordDict[EAR_THREE] = earThree
    coordDict[EAR_FOUR] = earFour
    coordDict[EAR_FIVE] = earFive

    return coordDict

def getMinMax(coordDict):
    xs = np.absolute(np.array([coordDict[LEFT_EYE][0], 
                               coordDict[RIGHT_EYE][0], 
                               coordDict[EAR_ONE][0], 
                               coordDict[EAR_TWO][0], 
                               coordDict[EAR_THREE][0], 
                               coordDict[EAR_FOUR][0], 
                               coordDict[EAR_FIVE][0]]).astype(np.float))
    ys = np.absolute(np.array([coordDict[LEFT_EYE][1], 
                               coordDict[RIGHT_EYE][1], 
                               coordDict[EAR_ONE][1], 
                               coordDict[EAR_TWO][1], 
                               coordDict[EAR_THREE][1], 
                               coordDict[EAR_FOUR][1], 
                               coordDict[EAR_FIVE][1]]).astype(np.float))

    topX = int(np.amax(xs))
    topY = int(np.amax(ys))
    botX = int(np.amin(xs))
    botY = int(np.amin(ys))

    return topX, topY, botX, botY


def getIsolatedRotatedFace(fullPath, catGray):

    coordDict = getCoordInfo(fullPath)
    topX, topY, botX, botY = getMinMax(coordDict)

    angleToRotate = findAngle(coordDict[LEFT_EYE], coordDict[RIGHT_EYE])
    newRotatedImage = rotate(catGray, angleToRotate, center=coordDict[LEFT_EYE])

    newBotRightY, newBotRightX = newRotate((coordDict[LEFT_EYE][1], coordDict[LEFT_EYE][0]), (topY, botX), math.radians(angleToRotate))
    newBotRightY = max(newBotRightY, 0)
    newBotRightX = max(newBotRightX, 0)
    newBotRightY = min(newBotRightY, catGray.shape[0])
    newBotRightX = min(newBotRightX, catGray.shape[1])

    newTopRightY, newTopRightX = newRotate((coordDict[LEFT_EYE][1], coordDict[LEFT_EYE][0]), (botY, topX), math.radians(angleToRotate))
    newTopRightY = max(newTopRightY, 0)
    newTopRightX = max(newTopRightX, 0)
    newTopRightY = min(newTopRightY, catGray.shape[0])
    newTopRightX = min(newTopRightX, catGray.shape[1])

    newTopY = max(coordDict[LEFT_EYE][1], newTopRightY, newBotRightY)
    newTopX = max(coordDict[LEFT_EYE][0], newTopRightX, newBotRightX)
    newBotY = min(coordDict[LEFT_EYE][1], newTopRightY, newBotRightY)
    newBotX = min(coordDict[LEFT_EYE][0], newTopRightX, newBotRightX)

    grayFace = newRotatedImage[newBotY: newTopY, newBotX: newTopX]

    return grayFace


def getResizedCannyFace(newGrayEdges):

    oRows, oCols = newGrayEdges.shape

    rowsToPad = 0
    colsToPad = 0
    paddedImage = np.copy(newGrayEdges)

    if(oRows > oCols):
        colsToPad = oRows - oCols
        if(colsToPad % 2 != 0):
            pad_width = ((1,0),(0,0))
            paddedImage = np.pad(paddedImage, pad_width, mode='edge')
            colsToPad += 1
        
        pad_width = ((0,0),(colsToPad//2,colsToPad//2))
        paddedImage = np.pad(paddedImage, pad_width, mode='edge')


    elif(oCols > oRows):
        rowsToPad = oCols - oRows
        if(rowsToPad % 2 != 0):
            pad_width = ((0,0),(1,0))
            paddedImage = np.pad(paddedImage, pad_width, mode='edge')
            rowsToPad += 1

        pad_width = ((rowsToPad//2,rowsToPad//2),(0,0))
        paddedImage = np.pad(paddedImage, pad_width, mode='edge')


    else:
        rowsToPad = 0




    resizedImage = resize(paddedImage, (TARGET_ROWS, TARGET_COLS), anti_aliasing=False)

    # Attempts to get rid of some blurring from resizing.
    adjusted = (resizedImage > 0.1)

    return resizedImage




def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    flippedKernel = np.copy(kernel)
    flippedKernel = np.flip(flippedKernel, 0)
    flippedKernel = np.flip(flippedKernel, 1)
        
    for cImageRow in range(Hi):
        for cImageCol in range(Wi):
            tempImageArray = padded[cImageRow : cImageRow + Hk, cImageCol : cImageCol + Wk]
            finalVal = np.sum(tempImageArray * flippedKernel)
            out[cImageRow][cImageCol] = finalVal  
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    frontConst = 1 / (2 * np.pi * (sigma ** 2))
    bottomConst = 2 * (sigma ** 2)
    k = size // 2 
    
    for i in range(size):
        for j in range(size):
            kernel[i][j] = frontConst * np.exp(-1 * ((i - k)**2 + (j - k)**2) / (bottomConst))
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    xKernel = np.array([0.5, 0, -0.5])
    xKernel = np.reshape(xKernel, (1, 3))
    out = conv(img, xKernel)
    # 1 0 -1
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    yKernel = np.array([0.5, 0, -0.5])
    yKernel = np.reshape(yKernel, (3, 1))
    out = conv(img, yKernel)
    # 1 
    # 0
    # -1
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = 4 * partial_x(img)
    Gy = partial_y(img)
    
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = (np.rad2deg((np.arctan2(Gy, Gx))) + 360) % 360
    ### END YOUR CODE

    return G, theta, Gx, Gy


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    """
    Compare the edge strength of the current pixel with the edge strength of the pixel in the positive and negative gradient 
    directions. For example, if the gradient direction is south (theta=90), compare with the pixels to the north and south.

    If the edge strength of the current pixel is the largest; preserve the value of the edge strength. If not, suppress (i.e. 
    remove) the value.
    
     pixel gradient is >= its neighbors 
    """
    for i in range(H):
        for j in range(W):
            currTheta = (theta[i, j]) % 180
            currVal = G[i, j]
            
            neighOneX = i
            neighOneY = j
            neighTwoX = i
            neighTwoY = j
            
            if(currTheta == 0):
                neighOneY+=1
                neighTwoY-=1
            elif(currTheta == 45):
                neighOneY+=1
                neighOneX+=1
                neighTwoY-=1
                neighTwoX-=1
            elif(currTheta == 90):
                neighOneX+=1
                neighTwoX-=1
            else:
                neighOneY-=1
                neighOneX+=1
                neighTwoY+=1
                neighTwoX-=1
                
            neighOneVal = G[neighOneX][neighOneY] if (neighOneX < H and neighOneY < W and neighOneX >= 0 and neighOneY >= 0) else 0 
            neighTwoVal = G[neighTwoX][neighTwoY] if (neighTwoX < H and neighTwoY < W and neighTwoX >= 0 and neighTwoY >= 0) else 0
                
            outVal = 0
            
            if G[i, j] >= neighOneVal and G[i][j] >= neighTwoVal:
                outVal = G[i][j]
            out[i][j] = outVal
                
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    H, W = img.shape

    for i in range(H):
        for j in range(W):
            currVal = img[i][j]
            
            if(currVal > high):
                strong_edges[i][j] = currVal
            if(currVal >= low and currVal < high):
                weak_edges[i][j] = currVal
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    foundPixels = []
    for i in range(H):
        for j in range(W):
            if(strong_edges[i][j] != 0): 
                pixelsToExplore = []
                pixelsToExplore.append((i, j))
                foundPixels.append((i, j))
                edges[i][j] = True
                
                while(len(pixelsToExplore) != 0):
                    currPairing = pixelsToExplore.pop()
                    currNeighborList = get_neighbors(currPairing[0], currPairing[1], H, W)
                    for currNeighbor in currNeighborList:  
                        if(weak_edges[currNeighbor[0]][currNeighbor[1]] != 0):
                            if(currNeighbor not in foundPixels):
                                foundPixels.append(currNeighbor)
                                pixelsToExplore.append(currNeighbor)
                                edges[currNeighbor[0]][currNeighbor[1]] = True
                            
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)

    G, theta, Gx, Gy = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)



    ### END YOUR CODE

    return edge, Gx, Gy, smoothed, nms, strong_edges, weak_edges

def isInt(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def getTrainableDataset(max_examples=-1):
    labelsPath = "./labels"
    imagesPath = "./cats"
    csvFiles = []
    for _, _, files in os.walk(labelsPath):
        if files is not None:
            for file in files:
                csvFiles.append(labelsPath + "/" + file)

    y_labs = []
    x_files = []
    for csvFile in csvFiles:
        with open(csvFile) as activeCSV:
            readCSV = csv.reader(activeCSV, delimiter=',')
            for row in readCSV:
                if row[1] and isInt(row[1]):
                    y_labs.append(int(row[1]))
                    folder = csvFile.split('/')[-1].split('.')[0]
                    dataPath = imagesPath + "/" + folder + "/" + row[0]
                    x_files.append(dataPath)

    num_examples = 0
    num_features = None
    X_builder = None
    y_builder = np.asarray(y_labs, dtype=int)
    for file in x_files:
        imageMat = io.imread(file)
        features = imageToFeatures(imageMat, file)
        if num_features == None:
            num_features = features.shape[0]
            X_builder = np.zeros((1, num_features))
            X_builder[0, :] = features
        else:
            X_builder = np.vstack((X_builder, features))
        num_examples += 1
        print("File " + str(num_examples) + ": " + file)
        if num_examples == max_examples:
            y_builder = y_builder[:max_examples]
            break
    return X_builder, y_builder

