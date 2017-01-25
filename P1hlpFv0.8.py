import math, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import namedtuple
from subprocess import call

def grayscale(img, saveF=''):
    """Applies the Grayscale transform: Ret - img w/ only one color channel"""
    #plt.imshow(gray, cmap='gray')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(saveF) > 0: mpimg.imsave(gray_img)
    return gray_img
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else: ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #print (lines.shape)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, thickn=2):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=thickn)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
       `initial_img` should be the image before any processing.
    The result image is computed as follows:
        initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def o_process_image(image, iDir='',  iFnm='', saveFs=0, retMode='m'):
    u_image = (image*255).astype('uint8')
    gray    = grayscale(u_image)    
    #plt.imshow(gray, cmap='gray')
    blur_g  = gaussian_blur(gray, kernel_size=5)
    edges   = canny(blur_g, low_threshold=50, high_threshold=150)
    imshape  = image.shape
    Verts    = namedtuple('Verts', ['lT', 'lB', 'rT', 'rB'])
    Top = 330
    V = Verts( (0,imshape[0]),(450, Top), (550, Top), (imshape[1],imshape[0]) )
    m_edges  = region_of_interest(edges, vertices = np.array([[V.lT, V.lB, V.rT, V.rB ]], dtype=np.int32))
    ln_img   = hough_lines(m_edges, rho=2, theta=np.pi/180, threshold=20, min_line_len=40, max_line_gap=20)
    proc_img = np.dstack( (edges, edges, edges) )
    if   retMode == 'm': ln_edges = weighted_img(image, ln_img, α=0.8, β=1., λ=0.)
    elif retMode == 'l': ln_edges = weighted_img(prc_img, ln_img, α=0.8, β=1., λ=0.)
    else: ln_edges = weighted_img(np.zeros((imshape[0],imshape[1], 3) ).astype('uint8'), 
                                  ln_img, α=0.8, β=1., λ=0.)
    return ln_edges

def SaveOutF(write_img, inFnm='', imgDir=''):
    outFnm = imgDir+inFnm[:-4]+'_o.png'
    mpimg.imsave(outFnm, write_img)
    #print('Wrote File: '+outFnm)

def runImages(imgF):
    imgD   = ''
    in_image = mpimg.imread(imgD+imgF)
    outEdges = process_image(in_image, iDir=imgD, iFnm=imgF, retMode='')
    SaveOutF(outEdges, inFnm=imgD+imgF)
    plt.imshow(outEdges)
    #plt.show()

def process_image(image, iDir='',  iFnm='', saveFs=0):
    u_image = (image*255).astype('uint8')
    gray    = grayscale(u_image)    
    #plt.imshow(gray, cmap='gray')
    blur_g  = gaussian_blur(gray, kernel_size=5)
    edges   = canny(blur_g, low_threshold=50, high_threshold=150)
    imshape  = image.shape
    Verts    = namedtuple('Verts', ['lB', 'lT', 'rT', 'rB'])

    Top = 330
    Btm = imshape[0]-60

    lV  = Verts( (0,Btm),(450, Top), (500, Top), (550,Btm) )
    rV  = Verts( (500,Btm),(500, Top), (550, Top), (imshape[1],Btm) )
    lm_edges  = region_of_interest(edges, vertices = np.array([[lV.lB, lV.lT, lV.rT, lV.rB ]], dtype=np.int32))
    rm_edges  = region_of_interest(edges, vertices = np.array([[rV.lB, rV.lT, rV.rT, rV.rB ]], dtype=np.int32))
    ρ = 2
    θ = (np.pi)/180
    threshold=50
    min_ln_len=10
    max_ln_gap=100
    thickn=20
    #lln_img   = hough_lines(lm_edges, rho=2, theta=np.pi/180, threshold=20, min_line_len=40, max_line_gap=20)
    lln_img   = hough_lines(lm_edges, ρ, θ, threshold, min_ln_len, max_ln_gap, thickn)
    rln_img   = hough_lines(rm_edges, ρ, θ, threshold, min_ln_len, max_ln_gap, thickn)
    ln_img = lln_img + rln_img
    proc_img = np.dstack( (edges, edges, edges) )
    #ln_edges = weighted_img(np.zeros((imshape[0],imshape[1], 3) ).astype('uint8'), ln_img, α=0.8, β=1., λ=0.)
    ln_edges = weighted_img(image, ln_img, α=0.8, β=1., λ=0.)
    return ln_edges

for imgF in glob.glob("test_images/*.jpg"):
    #continue
    imgD   = ''
    in_image = mpimg.imread(imgD+imgF)
    outEdges = process_image(in_image)
    SaveOutF(outEdges, inFnm=imgD+imgF)
    plt.imshow(outEdges)
    plt.show()

if 1:
    white_output = './white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    call(["vlc", white_output])
if 1:
    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)
    call(["vlc", yellow_output])
if 1:
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)
    call(["vlc", challenge_output])
