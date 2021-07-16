import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def make_normal_albedo(radius, albedo_value):
    len = 2 * radius + 1
    cen = radius
    normal = np.zeros((len, len, 3))
    albedo = np.full((len, len), albedo_value)
    mask = np.empty((len, len), dtype=bool)

    for i in range(len):
        for j in range(len):
            mask[i, j] = ((i-cen)**2 + (j-cen)**2 <= radius**2)
            kx = (j-cen) / radius
            ky = (cen-i) / radius
            kz = sqrt(max(1 - kx**2 - ky**2, 0))
            normal[i, j, 0] = kx
            normal[i, j, 1] = ky
            normal[i, j, 2] = kz

    albedo[~mask] = 0
    normal[~mask] = 0

    return (albedo, normal, mask)


def measurement(light, albedo, normal):
    B = (albedo * normal.T).T
    measurement = B @ light
    index = measurement < 0
    measurement[index] = 0

    return measurement


if __name__ == '__main__':
    albedo, normal, mask = make_normal_albedo(200, 1)
    light = np.array([0, 1, 0])
    M = measurement(light, albedo, normal)

    plt.figure(12)

    plt.subplot(221)
    plt.title('albedo')
    plt.imshow(albedo, cmap='gray')

    plt.subplot(222)
    plt.title('normal')
    # RGB to GBR
    normal_img = normal.copy()
    #normal_img[:, :, 0] = normal[:, :, 2]
    #normal_img[:, :, 2] = normal[:, :, 0]
    normal_img = (normal_img + 1.0) / 2.0
    normal_img[~mask] = 0
    plt.imshow(normal_img)

    plt.subplot(212)
    plt.title('measurement')
    plt.imshow(M, cmap='gray')

    plt.imsave('albedo.png', albedo, cmap='gray')
    plt.imsave('normal.png', normal_img)
    plt.imsave('measurement.png', M, cmap='gray')
    plt.savefig('./fig.png')