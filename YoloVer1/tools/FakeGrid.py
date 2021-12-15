import torch


def make_grid(conf=1., bbox=(0., 0., 1., 1.), obj=-1):
    t = torch.zeros(1 + 4 + 10)

    # set the confidence
    t[0] = conf

    # set the bounding box
    t[1] = bbox[0]
    t[2] = bbox[1]
    t[3] = bbox[2]
    t[4] = bbox[3]

    # set the object classes
    if obj >= 0:
        t[5 + obj] = 1

    return t
