import cv2
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import glob
import os.path as op

def smoothedGrasient(I, ks, sig):
    # gradient (h, w, c)
    # padding: mirror (h+2, w+2, c)
    I_p = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    # filter: [-1, 0, 1] (h, w, c)
    g_x = (I_p[2:, 1:-1] - I_p[:-2, 1:-1]) / 2
    g_y = (I_p[1:-1, 2:] - I_p[1:-1, :-2]) / 2
    
    # blurred gradient (h, w, c)
    I_x = cv2.GaussianBlur(g_x, ks, sig)
    I_y = cv2.GaussianBlur(g_y, ks, sig)
    
    return I_x, I_y

def localMaximum(I, threshold, ks=(3, 3)):
    # get coords of value greater then threshold
    coords = np.argwhere(I > threshold)
    print(coords.shape)
    
    # padding: mirror (h+2, w+2, c)
    I_p = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    
    # collect local-maxima
    local_maxima = []
    for coord in coords:
        # plus 1 for padding
        x, y = coord + 1
        
        if np.all(I_p[x, y] - I_p[x-1:x+2, y-1:y+2] >= 0):
            local_maxima.append(coord)
                                
    return np.array(local_maxima)

def subPixelRefinement(I, coords):
    
    # padding: mirror (h+2, w+2, c)
    I_p = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    
    sp_coords = []
    values = []
    for coord in coords:
        # plus 1 for padding
        x, y = coord + 1
        
        # 1st order derivative (gradient)
        g = np.array([[I_p[x+1, y] - I_p[x-1, y]], [I_p[x, y+1] - I_p[x, y-1]]]) / 2
        # 2nd order derivative (Hassian matrix)
        h_a = I_p[x-1, y] - 2*I_p[x, y] + I_p[x+1, y]
        h_b = I_p[x, y-1] - 2*I_p[x, y] + I_p[x, y+1]
        h_c = (I_p[x-1, y-1] - I_p[x-1, y+1] - I_p[x+1, y-1] + I_p[x+1, y+1]) / 4
        H = np.array([[h_a, h_c], [h_c, h_b]])
        
        # accurate maximum location
        dcoord_m = -np.matmul(la.inv(H), g)
        
        # TODO: change sampled pixel
        #if np.abs(dcoord_m[0, 0]) > 0.5:
        #    print('change x', dcoord_m[0, 0])
        #if np.abs(dcoord_m[1, 0]) > 0.5:
        #    print('change y', dcoord_m[1, 0])
        
        # maximum
        value = I_p[x, y] + 1/2 * np.matmul(g.transpose(), dcoord_m)
        
        # collect result
        sp_coords.append(coord + np.reshape(dcoord_m, 2))
        values.append(value[0, 0])

    return np.array(sp_coords), np.array(values)

def Harris(I, ks=(5, 5), sig_i=1.5, sig_d=1.0):
    # convert to float (0 ~ 1)
    I_f = I.astype(np.float32) / 255.0
    
    # smoothed gradient (h, w, c)
    I_x, I_y = smoothedGrasient(I_f, ks, sig_d)
    
    # (h, w, c)
    S_x2 = cv2.GaussianBlur(I_x * I_x, ks, sig_i)
    S_y2 = cv2.GaussianBlur(I_y * I_y, ks, sig_i)
    S_xy = cv2.GaussianBlur(I_x * I_y, ks, sig_i)
    
    # Harris Matrix (h, w, 2, 2, c)
    #H_l = np.moveaxis([[S_x2, S_xy], [S_xy, S_y2]], (0, 1), (2, 3))
    
    # determinant and trace (h, w, c)
    #det_H = H_l[:, :, 0, 0]*H_l[:, :, 1, 1] - H_l[:, :, 1, 0]*H_l[:, :, 0, 1]
    #tr_H = np.trace(H_l, axis1=2, axis2=3)
    det_H = S_x2 * S_y2 - S_xy * S_xy
    tr_H = S_x2 + S_y2
    
    return det_H, tr_H

def HarrisCornerDetector(img, k=0.05, th=1):
    '''
    Arguments:
        img: gray-scale image, 0~255
    '''
    
    # Harris (h, w, c)
    det_H, tr_H = Harris(img)
    
    # response (h, w, c)
    R = det_H - k*np.power(tr_H, 2)
    
    # thresholding & find local-maxima
    threshold = 0.01 * R.max()
    local_maxima = localMaximum(R, threshold)
    
    # non-maximum suppression
    #ANMS()
    
    # sub-pixel refinement
    key_points, _ = subPixelRefinement(R, local_maxima)
    
    return key_points, local_maxima

def simpleDescriptor(I, key_points):
    '''
    Arguments:
        img: gray-scale image, 0~255
    '''
    # convert to float (0 ~ 1)
    I_f = I.astype(np.float32) / 255.0
    h, w = I_f.shape
    
    # FIXME: the padding is to prevent accessing pixels out of image
    # however, this is not good since this may cause mismatch
    # the better way is to remove such key point
    # padding: mirror (h+2, w+2, c)
    #I_p = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    
    # compute descriptor
    descriptors = []
    new_keypoints = []
    for kp in key_points:
        x = int(round(kp[0]))
        y = int(round(kp[1]))
        
        # skip key point accessing invalid pixel
        if x < 1 or x > h-2 or y < 1 or y > w-2:
            continue
        
        # take 9x9 as descriptor
        des = np.reshape(I_f[x-1:x+2, y-1:y+2], 9)
        
        descriptors.append(des)
        new_keypoints.append(kp)
        
    return np.array(new_keypoints), np.array(descriptors)

def MSOPDescriptor(I, key_points, ks=(5, 5), sig_p=1.0, sig_o=4.5):
    '''
    Arguments:
        img: gray-scale image, 0~255
    '''
    # convert to float (0 ~ 1)
    I_f = I.astype(np.float32) / 255.0
    h, w = I_f.shape
    
    # smoothed gradient (h, w, c)
    I_x, I_y = smoothedGrasient(I_f, ks, sig_o)
    
    # major orientation (h, w, c)
    u_mag = la.norm([I_x, I_y], axis=0)
    # FIXME: u_mag might be zero
    cos_theta = I_x / u_mag
    sin_theta = I_y / u_mag
    
    # blur image (h, w, c)
    I_b = cv2.GaussianBlur(I_f, ks, 2*sig_p)
    
    # compute descriptor
    descriptors = []
    new_keypoints = []
    for kp in key_points:
        # major orientation of the key point
        # FIXME: what coord to use to get orientation?
        mo_cos = cos_theta[int(round(kp[0])), int(round(kp[1]))]
        mo_sin = sin_theta[int(round(kp[0])), int(round(kp[1]))]
        
        # sample 8x8 pixels
        des = []
        isInvalid = False
        for i in range(-4, 4):
            for j in range(-4, 4):
                # sample point coordinate
                x = kp[0] + 5*(i+0.5) * mo_cos
                y = kp[1] + 5*(j+0.5) * mo_sin
                
                # bilinear interpolation
                r_x = int(x)
                r_y = int(y)
                
                # skip key point accessing invalid pixel 
                isInvalid = r_x < 0 or r_x > h-2 or r_y < 0 or r_y > w-2
                if isInvalid:
                    break
                
                dx = x - r_x
                dy = y - r_y
                value = np.matmul(np.matmul([1-dx, dx], I_b[r_x:r_x+2, r_y:r_y+2]), [[1-dy], [dy]])
                        
                des.append(value[0])
            
            # skip key point accessing invalid pixel 
            if isInvalid:
                break
                
        # skip key point accessing invalid pixel 
        if isInvalid:
            continue
                
        # normalization
        des = (des - np.mean(des)) / np.std(des)
        
        # TODO: wavelet transform
        
        descriptors.append(des)
        new_keypoints.append(kp)
        
    return np.array(new_keypoints), np.array(descriptors)

def feature_extraction(img, des='MSOP'):
    '''
    Arguments:
        img: gray-scale, 0~255
        des: descriptor, either MSOP or simple
    '''
    # Harris corner detector
    key_points, local_maxima = HarrisCornerDetector(img_gray[i])

    if des == 'MSOP':
        # MSOP descriptor
        key_points, descriptors = MSOPDescriptor(img_gray[i], key_points)
    else:
        # naive descriptor
        key_points, descriptors = simpleDescriptor(img_gray[i], key_points)

    return key_points, descriptors


def imshow(img, cmap=None):
    plt.imshow(img, cmap)
    plt.show()

def load_data(data_path, data_ext):
    images = []
    for filename in sorted(glob.glob(op.join(data_path, '*.'+data_ext))):
        print(filename)
        
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        #imshow(img)
    
    return np.array(images)

def drawPoints(img, coords, coords2=None, color=[255, 0, 0], color2=[0, 0, 255]):
    img_c = np.copy(img)
    
    if coords2 is not None:
        for x, y in coords2:
            cv2.circle(img_c, (int(round(y)), int(round(x))), 2, color2, -1)
    
    for x, y in coords:
        cv2.circle(img_c, (int(round(y)), int(round(x))), 2, color, -1)

    imshow(img_c)



if __name__ == "__main__":
    # load data
    img_set = load_data('images/lab 2', 'JPG')

    # convert to gray-scale
    img_gray = []
    for img in img_set:
        img_gray.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        
    img_gray = np.array(img_gray)

    for i in range(len(img_set)):
        # Harris corner detector
        key_points, local_maxima = HarrisCornerDetector(img_gray[i])
        print(key_points.shape)
        
        # naive descriptor
        #key_points, descriptors = simpleDescriptor(img_gray[i], key_points)
        
        # MSOP descriptor
        key_points, descriptors = MSOPDescriptor(img_gray[i], key_points)
        
        print(key_points.shape)
        print(descriptors.shape)
        
        drawPoints(img_set[i], key_points, local_maxima)
    