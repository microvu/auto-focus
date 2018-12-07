import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

IMAGE_SIZE = 204


def load_focus_pass(file):
    data = np.fromfile('./img/' + file + '.dat', dtype='u1', count=-1, sep='')
    return np.reshape(data, [-1, IMAGE_SIZE, IMAGE_SIZE] )[::1] / 255

def combine_imgs(imgs, w, vmax):
    for img in imgs:
        img[:,0] = np.ones(img.shape[0]) * vmax
        img[0,:] = np.ones(img.shape[1]) * vmax
    t = np.array([np.hstack(imgs[i:i+w]) for i in range(0,len(imgs)-w+1,w)])
    return np.vstack(t)

def show(file):
    imgs = load_focus_pass(file)

    edges = []

    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    for img in imgs:
        x = ndimage.convolve(img,kx)
        y = ndimage.convolve(img,ky)
        # edges.append(np.hypot(x,y))
        edges.append(np.square(x) + np.square(y)) # InSpec uses the square of the gradient magnitude

    vmax = np.max([np.max(np.reshape(edge, -1)) for edge in edges])

    edges = np.array(edges)
    original = combine_imgs(imgs,8,1)
    res = combine_imgs(edges,8,vmax)
    slices = np.array([
        edges[:,70,:],
        edges[:,110,:],
        edges[:,160,:]
    ])
    slc = combine_imgs(slices,1,vmax)


    plt.imshow(original, cmap='gray', interpolation=None, vmax=1)
    # plt.savefig('./out/' + file + '.png', format='png', dpi=400)
    plt.show()

    plt.imshow(res, cmap='gray', interpolation=None)
    # plt.savefig('./out/' + file + '_edge.png', format='png', dpi=400)
    plt.show()

    plt.imshow(slc, cmap='gray', interpolation=None)
    # plt.savefig('./out/' + file + '_slice.png', format='png', dpi=400)
    plt.show()


# kernels applied to calculate the gradient in x and y
kx = np.array([[1,-1]])
ky = np.array([[1],[-1]])


show('cal_prof_z7')
show('post_coax_z7')
show('post_ring_z7')
show('post_ring_z0')
show('ruler_ring_z2')