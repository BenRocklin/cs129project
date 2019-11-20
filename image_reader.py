import numpy as np
from collections import defaultdict
import os
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

def imageToFeatures(imageMatrix):
	grayscale = rgb2gray(imageMatrix)
	resizedImage = resize(grayscale, (375, 500))

    ks = 5
    sig = 1.4
    h = 0.07
    l = 0.04

    grayEdges, GxGray, GyGray, smoothed, nms, strong_edges, weak_edges = canny(resizedImage, kernel_size=ks, sigma=sig, high=h, low=l)

    return grayEdges


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



def getTrainableDataset():
    return None


