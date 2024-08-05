from scipy.fftpack import dct, idct
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import cv2



q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

q_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
          [18, 21, 26, 66, 99, 99, 99, 99],
          [24, 26, 56, 99, 99, 99, 99, 99],
          [47, 66, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99]])

q_fat100 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1]])


    
def splitRGB(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    return r, g, b

def unsplitRGB(R, G, B):
    nl, nc = R.shape
    img = np.zeros((nl, nc, 3), dtype=np.uint8)

    img[:,:,0] = R
    img[:,:,1] = G
    img[:,:,2] = B
    return img


def showImg(img, caption = "", cmap = None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption)
    plt.show()


def colorMap(cor,matriz):
    return clr.LinearSegmentedColormap.from_list(cor, matriz, N=256)

def ex3(img):

    #Color maps
    cm_red = colorMap("red", [(0,0,0), (1,0,0)]) 
    cm_green = colorMap("green", [(0,0,0), (0,1,0)])  
    cm_blue = colorMap("blue", [(0,0,0), (0,0,1)]) 
    cm_grey = colorMap("grey",[(0,0,0), (1,1,1)])

    R, G, B = splitRGB(img)
    #showImg(R, "RED", cm_red)
    #showImg(G, "GREEN", cm_green)
    #showImg(B, "BLUE", cm_blue)
    #showImg(unsplitRGB(R, G, B), "RGB", cm_grey)
    unsplitRGB(R, G, B)


def image_padding(img):
    altura, largura, __ = img.shape
    altura_pad = 32 - (altura % 32) if altura % 32 != 0 else 0
    largura_pad = 32 - (largura % 32) if largura % 32 != 0 else 0
    img_pad = np.pad(img, ((0, altura_pad), (0, largura_pad), (0, 0)), mode = 'edge')
    return img_pad

def remove_padding(img_padded, original_shape):
    altura, largura, __ = original_shape
    img_original = img_padded[:altura, :largura,:]
    return img_original

def ex4(img):
    #print("Original image shape: ",img.shape)
    img_bm_padded = image_padding(img)
    #print("Padded image shape: ",img_bm_padded.shape)
    img_bm_no_padding = remove_padding(img_bm_padded, img.shape)
    #print("Image with padding removed shape: ",img_bm_no_padding.shape)
    #print("Removal correct? " , np.array_equal(img_bm_no_padding, img))
    return img_bm_padded


def RBGtoYCbCr(R,G,B):
    YcbCr_matriz = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    Y = R*YcbCr_matriz[0][0] + G*YcbCr_matriz[0][1] + B*YcbCr_matriz[0][2]
    Cb = R*YcbCr_matriz[1][0] + G*YcbCr_matriz[1][1] + B*YcbCr_matriz[1][2] + 128 
    Cr = R*YcbCr_matriz[2][0] + G*YcbCr_matriz[2][1] + B*YcbCr_matriz[2][2] + 128
    return Y, Cb, Cr

def YCbCrtoRGB(Y,Cb,Cr):
    RGB_matriz = np.linalg.inv([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    R = Y * RGB_matriz[0][0] + (Cb - 128) * RGB_matriz[0][1] + (Cr - 128) * RGB_matriz[0][2]
    G = Y * RGB_matriz[1][0] + (Cb - 128) * RGB_matriz[1][1] + (Cr - 128) * RGB_matriz[1][2]
    B = Y * RGB_matriz[2][0] + (Cb - 128) * RGB_matriz[2][1] + (Cr - 128) * RGB_matriz[2][2]
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    R = np.round(R).astype(np.uint8)
    G = np.round(G).astype(np.uint8)
    B = np.round(B).astype(np.uint8)
    return R,G,B

def ex5(img):
    cm_grey = colorMap("grey", [(0,0,0), (1,1,1)])  
    R,G,B = splitRGB(img)
    Y, Cb, Cr = RBGtoYCbCr(R,G,B)
    #showImg(Y, "Y", cm_grey)
    #showImg(Cb, "Cb", cm_grey)
    #showImg(Cr, "Cr", cm_grey)
    return Y, Cb, Cr

def ex5_2(Y,Cb,Cr,img,cm_grey, quality):
    R, G, B = YCbCrtoRGB(Y, Cb, Cr)
    recovered_img = unsplitRGB(R, G, B)

    recovered_img = remove_padding(recovered_img, img.shape)

    #print("Original pixel value [0, 0]:", img[0, 0])
    #print("Value retrieved from pixel [0, 0]:", recovered_img[0, 0])

    showImg(recovered_img, "Recovered RGB Image Quality = " + str(quality), cm_grey)

    return recovered_img



def downsampling_linear(Y, Cb, Cr, fator):
    if fator == 2:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
    elif fator == 0:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return Y, Cb_d, Cr_d


def downsampling_cubic(Y, Cb, Cr, fator):
    if fator == 2:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=1, interpolation=cv2.INTER_CUBIC)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=1, interpolation=cv2.INTER_CUBIC)
    elif fator == 0:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return Y, Cb_d, Cr_d


def upsampling_linear(Y_d, Cb_d, Cr_d, fator):
    if fator == 2:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
    elif fator == 0:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return Y_d, Cb, Cr


def upsampling_cubic(Y_d, Cb_d, Cr_d, fator):
    if fator == 2:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
    elif fator == 0:
        Cb = cv2.resize(Cb_d, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        Cr = cv2.resize(Cr_d, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Y_d, Cb, Cr

def ex6_3(Y, Cb, Cr):
    
    Y_d, Cb_d, Cr_d = downsampling_linear(Y, Cb, Cr, 2)

    #showImg(Y_d, "Y (downsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))
    #showImg(Cb_d, "Cb (downsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))
    #showImg(Cr_d, "Cr (downsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))

    return Y_d, Cb_d, Cr_d


def ex6_4(Y_d, Cb_d, Cr_d):
    Y, Cb, Cr = upsampling_linear(Y_d, Cb_d, Cr_d, 2)
    #showImg(Y, "Y (upsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))
    #showImg(Cb, "Cb (upsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))
    #showImg(Cr, "Cr (upsampling)", colorMap("grey",[(0,0,0), (1,1,1)]))
    return Y, Cb, Cr

 #exercicio 7
def dct_array(channel):
    return dct(dct(channel, norm="ortho").T, norm='ortho').T

def idct_array(dct_channel_arr):
    ans = idct(idct(dct_channel_arr, norm="ortho").T, norm='ortho').T
    return ans

def dct_image(y_d, cb_d, cr_d):
    dct_y = dct_array(y_d)
    dct_cb = dct_array(cb_d)
    dct_cr = dct_array(cr_d)
    return dct_y, dct_cb, dct_cr

def idct_image(dct_y, dct_cb, dct_cr):
    idct_y = idct_array(dct_y)
    idct_cb = idct_array(dct_cb)
    idct_cr = idct_array(dct_cr)
    return idct_y, idct_cb, idct_cr


def ex7_1(y_d, cb_d, cr_d, cmGray):
    dct_y, dct_cb, dct_cr = dct_image(y_d, cb_d, cr_d)
    dcts = {"Y": dct_y, "Cb": dct_cb, "Cr": dct_cr}
    idct_y, idct_cb, idct_cr = idct_image(dct_y, dct_cb, dct_cr)
    idcts = {"Y": idct_y, "Cb": idct_cb, "Cr": idct_cr}
    #for name, channel in dcts.items():
    #    fig = plt.figure()
    #    plt.title(f"{name}_DCT") #dct - log(abs(x)+0.0001)
    #    sh = plt.imshow(np.log(np.abs(channel) + 0.0001),cmGray)
    #    fig.colorbar(sh)
        #plt.show()

    eqs1 = [idct_y, idct_cb, idct_cr]
    eqs2 = [y_d, cb_d, cr_d]
    equals = 0
    for i in range(len(eqs1)):
        pixels = np.count_nonzero(np.abs(eqs1[i] - eqs2[i]) > 0.000001)
        if pixels == 0:
            equals += 1
        #else:
            #print('No. of different pixels: ', np.count_nonzero(np.abs(eqs1[i] - eqs2[i]) > 0.000001))
            #showImg(eqs1[i], cmGray)
            #showImg(eqs2[i], cmGray)
            #showImg(eqs2[i] - eqs1[i], cmGray)
    #print('No. of equal channels: ', equals)


def dct_channel_by_blocks(channel, bs):
    sh =channel.shape
    ans= np.zeros(channel.shape)
    for i in range(0,sh[0],bs):
        for j in range(0,sh[1],bs):
            portion = channel[i:i+bs, j:j+bs]
            ans[i:i+bs, j:j+bs] = dct_array(portion)
    return ans

def dct_by_blocks(y_d, cb_d, cr_d,bs):
    y_dct = dct_channel_by_blocks(y_d, bs)
    cb_dct= dct_channel_by_blocks(cb_d, bs)
    cr_dct= dct_channel_by_blocks(cr_d, bs)
    return y_dct, cb_dct, cr_dct

def idct_channel_by_blocks(dct_image, bs):
    sh =dct_image.shape
    ans= np.zeros(dct_image.shape)
    for i in range(0,sh[0],bs):
        for j in range(0,sh[1],bs):
            portion = dct_image[i:i+bs, j:j+bs]
            ans[i:i+bs, j:j+bs] = idct_array(portion)
    return ans

def idct_by_blocks(y_dct, cb_dct, cr_dct,bs):
    y_idct = idct_channel_by_blocks(y_dct, bs)
    cb_idct= idct_channel_by_blocks(cb_dct, bs)
    cr_idct= idct_channel_by_blocks(cr_dct, bs)
    return y_idct, cb_idct, cr_idct


def ex7_23(y_d, cb_d, cr_d, bs,cmGray):
    y_dct, cb_dct, cr_dct = dct_by_blocks(y_d, cb_d, cr_d, bs)
    arrplot= [('Yb_DCT',y_dct),('Cbb_DCT',cb_dct),('Crb_DCT',cr_dct)]
    #for s,p in arrplot:
        #showImg(np.log(np.abs(p)+0.0001),s, cmGray)
    y_idct, cb_idct,cr_idct =  idct_by_blocks( y_dct, cb_dct, cr_dct,bs)
    eqs1 = [y_idct,cb_idct,cr_idct]
    eqs2 = [y_d,cb_d,cr_d]
    equals=0
    for i in range(len(eqs1)):
        howmany=np.count_nonzero(np.abs(eqs1[i]-eqs2[i])>0.000001)
        if howmany==0:
            equals +=1
        #else:
            #print('no of different pixels: ',np.count_nonzero(np.abs(eqs1[i]-eqs2[i])>0.000001))
            #showImg(eqs1[i],cmGray)
            #showImg(eqs2[i],cmGray)
            #showImg(eqs2[i]-eqs1[i],cmGray)
    #print('No. of equal channels: ',equals)
    return y_dct, cb_dct, cr_dct


def calcQuality(quality):
            
    if quality >= 50:
        scaleFactor = (100 - quality) / 50
    else:
        scaleFactor = 50 / quality

    if scaleFactor == 0:
        qualityQ_Y = q_fat100
        qualityQ_CBCR = q_fat100
    else:
        qualityQ_Y = q_y * scaleFactor
        qualityQ_CBCR = q_cbcr * scaleFactor


    qualityQ_Y = np.clip(qualityQ_Y, 1, 255).astype(np.uint8)
    qualityQ_CBCR = np.clip(qualityQ_CBCR, 1, 255).astype(np.uint8)

    return qualityQ_Y, qualityQ_CBCR

def ex8_1(blocks, quality, Y, CB, CR, cm_grey):

    qualityQ_Y, qualityQ_CBCR = calcQuality(quality)

    length = Y.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice_Y = Y[i:i+blocks, j:j+blocks]
            Y[i:i+blocks, j:j+blocks] = slice_Y / qualityQ_Y

    #print(np.round(Y[8:16, 8:16]))                             

    length = CB.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice_CB = CB[i:i+blocks, j:j+blocks]
            CB[i:i+blocks, j:j+blocks] = slice_CB / qualityQ_CBCR

            slice_CR = CR[i:i+blocks, j:j+blocks]
            CR[i:i+blocks, j:j+blocks] = slice_CR / qualityQ_CBCR

    Y = np.round(Y).astype(int)
    CB = np.round(CB).astype(int)
    CR = np.round(CR).astype(int)

    #showImg(np.log(abs(Y) + 0.0001), 'Y Quantized', cm_grey)
    #showImg(np.log(abs(CB) + 0.0001), 'Cb Quantized', cm_grey)
    #showImg(np.log(abs(CR) + 0.0001), 'Cr Quantized', cm_grey)

    return Y, CB, CR

def ex8_2(blocks, quality,Y,CB,CR,cm_grey):
         
        qualityQ_Y, qualityQ_CBCR = calcQuality(quality)

        length = Y.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = Y[i:i+blocks, j:j+blocks]
                Y[i:i+blocks, j:j+blocks] = slice * qualityQ_Y

        #print(np.round(Y[8:16, 8:16]))   
                
        length = CB.shape
        for i in range(0, length[0], blocks):
            for j in range(0, length[1], blocks):
                slice = CB[i:i+blocks, j:j+blocks]
                CB[i:i+blocks, j:j+blocks] = slice * qualityQ_CBCR

                slice = CR[i:i+blocks, j:j+blocks]
                CR[i:i+blocks, j:j+blocks] = slice * qualityQ_CBCR

        Y  = Y.astype(float)
        CB = CB.astype(float)
        CR = CR.astype(float)

        #showImg(np.log(abs(Y) + 0.0001), 'Y Iquantization', cm_grey)
        #showImg(np.log(abs(CB) + 0.0001), 'Cb Iquantization', cm_grey)
        #showImg(np.log(abs(CR) + 0.0001), 'Cr Iquantization', cm_grey)

        return Y, CB, CR


def dpcm(Y_qdct, Cb_qdct, Cr_qdct, cmGray):


    Y_dpcm = np.copy(Y_qdct)
    Cb_dpcm = np.copy(Cb_qdct)
    Cr_dpcm = np.copy(Cr_qdct)

    for i in range(int(Y_qdct.shape[0] / 8)):
        for j in range(int(Y_qdct.shape[1] / 8)):
            if (i != 0):
                if (j != 0):
                    Y_dpcm[i * 8, j * 8] = Y_qdct[i * 8, j * 8] - Y_qdct[i * 8, j * 8 - 8]
                else:
                    Y_dpcm[i * 8, j * 8] = Y_qdct[i * 8, j * 8] - Y_qdct[i * 8 - 8, int(Y_qdct.shape[1]) - 8]
            else:
                if (j != 0):
                    Y_dpcm[i * 8, j * 8] = Y_qdct[i * 8, j * 8] - Y_qdct[i * 8, j * 8 - 8]

    for i in range(int(Cb_qdct.shape[0] / 8)):
        for j in range(int(Cb_qdct.shape[1] / 8)):
            if (i != 0):
                if (j != 0):
                    Cb_dpcm[i * 8, j * 8] = Cb_qdct[i * 8, j * 8] - Cb_qdct[i * 8, j * 8 - 8]
                    Cr_dpcm[i * 8, j * 8] = Cr_qdct[i * 8, j * 8] - Cr_qdct[i * 8, j * 8 - 8]
                else:
                    Cb_dpcm[i * 8, j * 8] = Cb_qdct[i * 8, j * 8] - Cb_qdct[i * 8 - 8, int(Cb_qdct.shape[1]) - 8]
                    Cr_dpcm[i * 8, j * 8] = Cr_qdct[i * 8, j * 8] - Cr_qdct[i * 8 - 8, int(Cb_qdct.shape[1]) - 8]
            else:
                if (j != 0):
                    Cb_dpcm[i * 8, j * 8] = Cb_qdct[i * 8, j * 8] - Cb_qdct[i * 8, j * 8 - 8]
                    Cr_dpcm[i * 8, j * 8] = Cr_qdct[i * 8, j * 8] - Cr_qdct[i * 8, j * 8 - 8]

    quantLogY = np.log(np.abs(Y_dpcm) + 0.0001)
    quantLogCb = np.log(np.abs(Cb_dpcm) + 0.0001)
    quantLogCr = np.log(np.abs(Cr_dpcm) + 0.0001)

    #showImg(quantLogY, "Yb_DPCM", cmGray)
    #showImg(quantLogCb,"Cbb_DPCM", cmGray)
    #showImg(quantLogCr, "Crb_DPCM", cmGray)

    return Y_dpcm, Cb_dpcm, Cr_dpcm


def reverse_dpcm(Y_dpcm, Cb_dpcm, Cr_dpcm):
    Y_qdct = np.copy(Y_dpcm)
    Cb_qdct = np.copy(Cb_dpcm)
    Cr_qdct = np.copy(Cr_dpcm)

    for i in range(int(Y_dpcm.shape[0] / 8)):
        for j in range(int(Y_dpcm.shape[1] / 8)):
            if (i != 0):
                if (j != 0):
                    Y_qdct[i * 8, j * 8] = Y_qdct[i * 8, j * 8 - 8] + Y_dpcm[i * 8, j * 8]
                else:
                    Y_qdct[i * 8, j * 8] = Y_qdct[i * 8 - 8, int(Y_dpcm.shape[1]) - 8] + Y_dpcm[i * 8, j * 8]
            else:
                if (j != 0):
                    Y_qdct[i * 8, j * 8] = Y_qdct[i * 8, j * 8 - 8] + Y_dpcm[i * 8, j * 8]


    for i in range(int(Cb_dpcm.shape[0] / 8)):
        for j in range(int(Cb_dpcm.shape[1] / 8)):
            if (i != 0):
                if (j != 0):
                    Cb_qdct[i * 8, j * 8] = Cb_qdct[i * 8, j * 8 - 8] + Cb_dpcm[i * 8, j * 8]
                    Cr_qdct[i * 8, j * 8] = Cr_qdct[i * 8, j * 8 - 8] + Cr_dpcm[i * 8, j * 8]
                else:
                    Cb_qdct[i * 8, j * 8] = Cb_qdct[i * 8 - 8, int(Cb_dpcm.shape[1]) - 8] + Cb_dpcm[i * 8, j * 8]
                    Cr_qdct[i * 8, j * 8] = Cr_qdct[i * 8 - 8, int(Cb_dpcm.shape[1]) - 8] + Cr_dpcm[i * 8, j * 8]
            else:
                if (j != 0):
                    Cb_qdct[i * 8, j * 8] = Cb_qdct[i * 8, j * 8 - 8] + Cb_dpcm[i * 8, j * 8]
                    Cr_qdct[i * 8, j * 8] = Cr_qdct[i * 8, j * 8 - 8] + Cr_dpcm[i * 8, j * 8]

    return Y_qdct, Cb_qdct, Cr_qdct


def stats(img,image_reconstructed):
        
        io = img.astype(np.float32)
        ir = image_reconstructed.astype(np.float32)
        shape  = np.shape(img)
    
        MSE = np.sum(pow(io - ir, 2))/(int(shape[0]) * int(shape[1]))
        RMSE = np.sqrt(MSE)
        
        P = np.sum(pow(io, 2))/(int(shape[0]) * int(shape[1]))
        SNR = np.log10(P/MSE) * 10
        PSNR = np.log10(pow(np.max(io), 2) / MSE) * 10

        return MSE, RMSE, SNR, PSNR


def showDiffY(y_original, y_rebuilt, cmGray, text='Diff image'):

    aux = abs(y_original - y_rebuilt)

    print("AVG_diff: ", np.mean(aux))
    print("Max_diff: ", np.max(aux))
    
    showImg(aux, text, cmGray)


    #Encoder
def encoder(img,quality,bloco):

    cmGray = colorMap("grey",[(0,0,0), (1,1,1)])

    ex3(img)

    image_padding = ex4(img)

    Y, Cb, Cr = ex5(image_padding)

    y_d, cb_d, cr_d = ex6_3(Y, Cb, Cr)

    ex7_1(y_d, cb_d, cr_d, cmGray)

    y_dct8, cb_dct8, cr_dct8 = ex7_23(y_d, cb_d, cr_d, 8, cmGray)

    y_dct64, cb_dct64, cr_dct64 = ex7_23(y_d, cb_d, cr_d, 64, cmGray)

    y_quant, cb_quant, cr_quant = ex8_1(bloco, quality, y_dct8, cb_dct8, cr_dct8, cmGray)

    dpcm_y, dpcm_cb, dpcm_cr = dpcm(y_quant, cb_quant, cr_quant, cmGray)

    return dpcm_y, dpcm_cb, dpcm_cr, cmGray, Y


#Decoder
def decoder(dpcm_y, dpcm_cb, dpcm_cr, cmGray, img, bloco, quality):

    y_quant, cb_quant, cr_quant = reverse_dpcm(dpcm_y, dpcm_cb, dpcm_cr)

    y_dct, cb_dct, cr_dct = ex8_2(bloco, quality, y_quant, cb_quant, cr_quant, cmGray)

    y_idct, cb_idct, cr_idct = idct_by_blocks(y_dct, cb_dct, cr_dct, bloco)

    Y, Cb, Cr = ex6_4(y_idct, cb_idct, cr_idct)

    image_reconstructed = ex5_2(Y,Cb,Cr,img,cmGray,quality)

    return image_reconstructed, Y


def main():

    #Read image
    imgs = ["airport.bmp", "geometric.bmp", "nature.bmp"]
    for i in range(len(imgs)):
        img = plt.imread("imagens/"+imgs[i])
        #print(img.shape)
        #print(img.dtype)
        #showImg(img, "IMAGEM" + fname)

        qualities = [10,25,50,75,100]

        for quality in qualities:

            print("Quality: ", quality)
            
            dpcm_y, dpcm_cb, dpcm_cr, cmGray, Y_Original = encoder(img,quality,8)
            image_reconstructed, Y_rebuilt = decoder(dpcm_y, dpcm_cb, dpcm_cr, cmGray, img, 8, quality)
            
            showDiffY(Y_Original, Y_rebuilt, cmGray, "Diff Image")

            MSE, RMSE, SNR, PSNR = stats(img, image_reconstructed)
            print("MSE: ", MSE)
            print("RMSE: ", RMSE)
            print("SNR: ", SNR)
            print("PSNR: ", PSNR, "\n")



if __name__ == '__main__':
    main()