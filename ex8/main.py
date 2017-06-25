import Queue as Q
import numpy as np
import utils

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
    return fg_unaries, bg_unaries


if __name__ == '__main__':
    # print "hello world!"

    # first fg, second bg
    domain = (0, 255, 0, 255)

    priority_queue = Q.PriorityQueue()

    img = utils.readImageAsGray('branchAndMinCut/garden.png')
    img_mean = np.mean(img)

    fg_mean, bg_mean = get_mean_for_domain(domain, img_mean)

    fg_unaries, bg_unaries = compute_unaries(img, fg_mean, bg_mean)

