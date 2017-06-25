import Queue as Q
import numpy as np
import utils
from matplotlib import pyplot as plt


LAMBDA1 = 0.0001
LAMBDA2 = 0.0001

MU = 1
NU = 0.1


def get_mean_value(img_mean, min_val, max_val):
    if img_mean < min_val:
        return min_val
    if img_mean > max_val:
        return max_val
    return img_mean


def get_mean_for_domain(current_domain, img_mean):
    fg_mean = get_mean_value(img_mean, current_domain[0], current_domain[1])
    bg_mean = get_mean_value(img_mean, current_domain[2], current_domain[3])
    return fg_mean, bg_mean


def compute_unaries(img, fg_mean, bg_mean):
    fg_unaries = NU + LAMBDA1 * np.square(img - fg_mean)
    bg_unaries = LAMBDA2 * np.square(img - bg_mean)
    return np.stack([fg_unaries, bg_unaries], axis=-1)


def get_energy(domain, img_mean, img):
    fg_mean, bg_mean = get_mean_for_domain(domain, img_mean)
    unaries = compute_unaries(img, fg_mean, bg_mean)

    return utils.gridGraphCut(unaries, MU)


def split_domain(domain):
    xs, xe, ys, ye = domain

    if xe - xs < ye - ys:
        mid = (ye + ys) // 2
        return (xs, xe, ys, mid), (xs, xe, mid, ye)

    mid = (xs + xe) // 2
    return (xs, mid, ys, ye), (mid, xe, ys, ye)


if __name__ == '__main__':
    # first fg, second bg
    domain = (0, 256, 0, 256)

    priority_queue = Q.PriorityQueue()

    img = utils.readImageAsGray('branchAndMinCut/garden.png')
    img = img.astype(np.float32)
    img_mean = np.mean(img)

    energy, _ = get_energy(domain, img_mean, img)

    priority_queue.put((energy, domain))

    while True:
        node = priority_queue.get()
        domain = node[1]
        if domain[0] == domain[1] - 1 and domain[2] == domain[3] - 1:
            _, isSource = get_energy(domain, img_mean, img)
            break

        domain1, domain2 = split_domain(domain)
        energy1, _ = get_energy(domain1, img_mean, img)
        energy2, _ = get_energy(domain2, img_mean, img)

        priority_queue.put((energy1, domain1))
        priority_queue.put((energy2, domain2))

    result_img = (domain[0] * isSource + domain[2] * (1 - isSource)).astype(np.uint8)
    plt.imshow(result_img, cmap='gray', vmin=0, vmax=255)
    plt.show()