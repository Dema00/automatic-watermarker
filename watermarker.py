import numpy as np
import cv2
import operator
import sys
import argparse

#ripped from https://github.com/andrewdcampbell/seam-carving/blob/master/seam_carving.py
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)     
        
    return energy

#find the darkest spot in the image
def findDark(im):
    imBlur = cv2.GaussianBlur(im,(109,109),cv2.BORDER_DEFAULT)
    cv2.imwrite("blur.jpg", imBlur)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imBlur)

    return minLoc

#add the watermark
def addWatermark(imageCoords, backgroundImg, watermark_name):

    watermark = cv2.imread(watermark_name, -1)
    assert watermark is not None

    #resize to scale
    scale_x, scale_y = watermark.shape[:2]
    imgSide = min(backgroundImg.shape[1],backgroundImg.shape[0])
    imgSide_scaled = imgSide/8
    resize_factor = scale_x/scale_y
    appropriate_scale = tuple((int(imgSide_scaled), int(imgSide_scaled*resize_factor)))
    watermark = cv2.resize(watermark, appropriate_scale)

    #since the coordinates are (ideally) the center of the darkest zone
    #we need to offset the image properly

    imageCoords = tuple(map(operator.sub, imageCoords,(watermark.shape[1]//2, watermark.shape[0]//2)))

    #top left position of overlay
    imageCoords = tuple(reversed(imageCoords)) #for some reason the tuple gets swapped, this fixes that

    top_left_y, top_left_x = imageCoords

    #bottom right position of overlay
    bottom_right_y, bottom_right_x = tuple(map(operator.add, imageCoords, (watermark.shape[0], watermark.shape[1])))


    #check if watermark is out of bounds
    if top_left_y < 0:
        bottom_right_y = bottom_right_y - top_left_y
        top_left_y = 0

    if top_left_x < 0:
        bottom_right_x = bottom_right_x - top_left_x

    #calculating alpha channel value

    alpha_overlay = watermark[:,:,3]/255 #normalizing alpha
    alpha_background = 1 - alpha_overlay

    for c in range(0, 3):
         backgroundImg[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c] = (
             alpha_overlay * watermark[:, :, c] + alpha_background * backgroundImg[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c])


    
    return backgroundImg

#Main function
def main():
    parser = argparse.ArgumentParser(description = "Automatic meme watermarking")

    argument = parser.add_argument("img_name" ,metavar="Image")
    argument = parser.add_argument("watermark_name" ,metavar="Watermark")
    args = parser.parse_args()

    im = cv2.imread(args.img_name)
    assert im is not None

    energy = forward_energy(im)
    cv2.imwrite("energy.jpg", energy)

    zone = findDark(energy)
    im_x, im_y = energy.shape

    radius = int(im_x/20)
    zone_x, zone_y = zone

    if zone_x < radius or zone_y < radius:
        zone = tuple(map(operator.add, zone, (int(radius), int(radius))))

    finalImage = addWatermark(zone, im, args.watermark_name)

    cv2.imwrite("Troll.jpg", finalImage)
    cv2.imwrite("Circle.jpg", im)

if __name__ == '__main__':
    sys.exit(main())