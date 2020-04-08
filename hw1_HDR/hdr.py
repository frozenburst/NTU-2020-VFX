import cv2
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

def imshow(img, cmap=None):
    plt.imshow(img, cmap)
    plt.show()

def load_data(data_path, img_type='.JPG'):
    img_set = []
    exposure_time = []

    with open(op.join(data_path, 'shutter_speed.csv'), 'r') as fp:
        lines = fp.readlines()

        for line in lines:
            filename, shutter_speed = line.split(', ')
            

            # exposure time
            shutter_speed = shutter_speed.split('\"')
            if len(shutter_speed) == 2:
                exp_t = int(shutter_speed[0]) + 0.1*int(shutter_speed[1])
            else:
                exp_t = 1/int(shutter_speed[0])
            exposure_time.append(exp_t)

            # read image
            filepath = op.join(data_path, filename + img_type)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_set.append(img)

            # display data
            print(filepath)
            print(exp_t)
            #imshow(img)
            
    img_set = np.array(img_set)

    return img_set, exposure_time

def random_sample(img_set, num_pixels):
    num_imgs, height, width, channels = img_set.shape

    x_coords = np.random.randint(0, height, (num_pixels, 1))
    y_coords = np.random.randint(0, width, (num_pixels, 1))
    coords = np.concatenate((x_coords, y_coords), axis=1)
    print(coords)

    sampled_pixels = [[[img_set[j][coords[i, 0], coords[i, 1]][c] for j in range(num_imgs)] for i in range(num_pixels)] for c in range(channels)]
    sampled_pixels = np.array(sampled_pixels)

    return sampled_pixels, coords

def downsampling(img_set, size, interpolation=cv2.INTER_LINEAR):
    channels = img_set.shape[3]
    sampled_pixels = []
    for img in img_set:
        ds_img = cv2.resize(img, size, interpolation=interpolation)
        ds_vec = np.moveaxis(np.reshape(ds_img, (-1, channels, 1)), 1, 0)
        sampled_pixels = np.concatenate(sampled_pixels, ds_vec)
    
    return sampled_pixels

def gsolve(Z, B, l, w):
    '''
    Assumes:
        Zmin = 0
        Zmax = 255

    Arguments:
        Z(c, i, j): sample pixel values at position i in image j of channel c
        B(j): log delta t(exposure time) of image j
        l: lambda, weighting regression term
        w(z): weighting function

    Returns:
        g(c, z): log exposure corresponding to pixel value z of chammel c
        lE(c, i): log irradiance at pixel position i of channel c
    '''

    g = []
    # lE = []

    # parameters
    n = 256 # intensity range
    channels, num_pixels, num_imgs = np.shape(Z)

    for c in range(channels):

        A = np.zeros((num_pixels * num_imgs + n + 1, n + num_pixels))
        b = np.zeros((A.shape[0], 1))

        k = 1 # current row of A
        for i in range(num_pixels):
            for j in range(num_imgs):
                # weight of current intensity
                wij = w[Z[c, i, j]]

                A[k, Z[c, i, j]] = wij
                A[k, n + i] = -wij
                b[k] = wij * B[j]

                k += 1

        # restrict 127 to be 0
        A[k, 127] = 1
        k += 1

        # smoothness term
        for i in range(n-2):
            lw = l * w[i+1]

            A[k, i] = lw
            A[k, i+1] = -2 * lw
            A[k, i+2] = lw

            k += 1
        
        # least-square solution for linear matrix equation
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        g.append(x[:n])
        # lE.append(x[n+1:])

    return np.reshape(g, (channels, n))

def Debevec_HDR(Z, B, P, l = 30, w=None):

    '''
    Arguments:
        Z(j, x, y, c): j images
        B(j): exposure of image j
        P(i, c): i sampled pixels
        l: lambda, weighting regression term
        w(z): weighting function

    Returns:
        lE(c, x, y): radiance map of channel c
        g(c, z): g function
    '''

    # parameters
    height, width, channels = Z[0].shape
    lE = np.zeros((channels, height, width))

    # log exprosure time
    lB = np.log(B)

    # hat weighting function
    if w is None:
        Zmin = 0
        Zmax = 255
        Zmed = (Zmin + Zmax)/2
        w = np.array([z-Zmin if z <= Zmed else Zmax-z for z in range(256)])/int(Zmed)

    # get function g
    g = gsolve(sampled_pixels, lB, l, w)

    '''
    # move axis version
    mZ = np.moveaxis(Z, [0, 3], [3, 0])
    for c in range(channels):
        wij = w[mZ[c]]
        lE[c] = np.sum(wij*(g[c, mZ[c]]-B), axis=-1) / np.sum(wij, axis=-1)
    
    '''
    # loop version
    progress = '0%'
    
    for c in range(channels):
        for x in range(height):
            for y in range(width):
                wij = w[Z[:, x, y, c]]
                swij = np.sum(wij)
                # check for overflow
                if swij == 0:
                    if np.all(Z[:, x, y, c] == 255):
                        lE[c, x, y] = np.inf
                    elif np.all(Z[:, x, y, c] == 0):
                        lE[c, x, y] = -np.inf
                    print(x, y, Z[:, x, y, c])
                else:
                    lE[c, x, y] = np.sum(wij * (g[c, Z[:, x, y, c]] - B)) / swij
            
            # display progress
            new_prog = '{}%'.format(int((c*height + x)/(channels*height) * 100))
            if new_prog != progress:
                progress = new_prog
                print(progress)

    # clip fix overflow
    for c in range(channels):
        lE[c, lE[c] == np.inf] = np.max(lE[c, lE[c] != np.inf])
        lE[c, lE[c] == -np.inf] = np.min(lE[c, lE[c] != -np.inf])

    # change from color major to row major
    lE = np.moveaxis(lE, 0, -1)
    lE = np.exp(lE)

    return lE, g


if __name__ == "__main__":
    # dataset
    data_path = op.join('.', 'images', 'nightsight')
    img_type = '.JPG'
    # load data
    img_set, exposure_time = load_data(data_path, img_type)

    # sample pixels
    sample_method = 'random_sample'
    if sample_method == 'random_sample':
        # random sample
        num_pixels = 50
        sampled_pixels, sampled_coords = random_sample(img_set, num_pixels)
    elif sample_method == 'downsampling':
        # downsampling
        sampled_pixels = downsampling(img_set, (10, 10))
    # check if the number of pixels is enough to recover the response curve 
    if sampled_pixels.shape[1] * (len(img_set)-1) < 256:
        print('The number of pixels is not enough!')

    # Debevec's method
    HDR_img, response = Debevec_HDR(img_set, exposure_time, sampled_pixels)
    print(HDR_img.shape)

    # save HDR image
    cv2.imwrite(op.join(data_path, 'radiance_map.hdr'), HDR_img)

    # heatmap of gray scale log space
    plt.clf()
    HDR_gray = np.sum(HDR_img, axis=2)/3
    plt.imshow(np.log(HDR_gray), 'rainbow')
    plt.colorbar()
    plt.savefig(op.join(data_path, 'log_exposure.png'))

    # save HDR image
    cv2.imwrite(op.join(data_path, 'radiance_map.hdr'), HDR_img)

    # save response curve
    plt.clf()
    y_range = np.arange(256)
    RGB = 'rgb'
    for i in range(3):
        plt.plot(response[i], y_range, color=RGB[i])
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.savefig(op.join(data_path, 'response_curve.png'))

    # save sampled pixel
    if sample_method == 'random_sample':
        plt.clf()
        plt.plot(sampled_coords[:, 1], sampled_coords[:, 0], 'ro')
        plt.imshow(img_set[-1], None)
        plt.savefig(op.join(data_path, 'sampled_pixel.png'))

    # Tone Mapping
    HDR_f32 = HDR_img.astype(np.float32)
    
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(HDR_f32)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(op.join(data_path, 'ldr.png'), ldr*255) 
    # drago
    tonemap = cv2.createTonemapDrago(2.2)
    ldr = tonemap.process(HDR_f32)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(op.join(data_path, 'ldr_drago.png'), ldr*255)
    # mantiuk
    tonemap = cv2.createTonemapMantiuk(2.2)
    ldr = tonemap.process(HDR_f32)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(op.join(data_path, 'ldr_mantiuk.png'), ldr*255)
    # reinhard
    tonemap = cv2.createTonemapReinhard(2.2)
    ldr = tonemap.process(HDR_f32)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(op.join(data_path, 'ldr_reinhard.png'), ldr*255)

