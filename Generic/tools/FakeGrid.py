import torch
import random

CONFIDENCES = 1
BOUNDING_BOXES = 1
OBJECT_CATEGORIES = 10


def target_grid(grid: torch.Tensor = None, conf=1., bbox=(0., 0., 1., 1.), obj=-1):
    if grid is None:
        grid = torch.zeros((CONFIDENCES + BOUNDING_BOXES * 4 + OBJECT_CATEGORIES))

    # set the confidence
    grid[:CONFIDENCES] = conf

    # set the bounding box
    grid[CONFIDENCES: CONFIDENCES + BOUNDING_BOXES * 4] = torch.tensor(bbox)

    # set the object classes
    if obj >= 0:
        grid[CONFIDENCES + BOUNDING_BOXES * 4:][obj] = 1

    return grid


def rand_grid(grid: torch.Tensor = None):
    if grid is None:
        grid = torch.zeros((CONFIDENCES + BOUNDING_BOXES * 4 + OBJECT_CATEGORIES))

    # set the confidence
    grid[:CONFIDENCES] = torch.rand(1)

    # set the bounding box
    grid[CONFIDENCES: CONFIDENCES + BOUNDING_BOXES * 4] = torch.rand(BOUNDING_BOXES * 4)

    # set the object classes
    obj = torch.randint(0, OBJECT_CATEGORIES, (1,))
    grid[CONFIDENCES + BOUNDING_BOXES * 4:][obj] = torch.rand(1)

    return grid


def gen_rand_set(rand_set_size, alpha=0, beta=1):
    """
    Generates a set of random numbers that do not repeat
    """

    temp_set = []

    while len(temp_set) < rand_set_size:

        rand_val = random.randint(alpha, beta - 1)

        if rand_val not in temp_set:
            temp_set.append(rand_val)

    return temp_set


def batch_grids_with_rand_vals(batch_size: int = 1, grids_size: int = 64, obj_count: int = 1):
    # create grids
    grids = torch.zeros((batch_size, CONFIDENCES + BOUNDING_BOXES * 4 + OBJECT_CATEGORIES, grids_size))

    # generate a set of random numbers that do not repeat
    rand_set = gen_rand_set(batch_size * obj_count, alpha=0, beta=grids_size)

    # generate grids
    for i in range(batch_size):
        for j in range(obj_count):

            # generate a random number
            rand_grid_val = rand_grid()

            # yield a generated grid to the grids
            grids[i, :, rand_set[i * obj_count + j]] = rand_grid_val

    return grids


def batch_grids_with_target_vals(batch_size: int = 1, grids_size: int = 64, obj_count: int = 1):

    # create grids
    grids = torch.zeros((batch_size, CONFIDENCES + BOUNDING_BOXES * 4 + OBJECT_CATEGORIES, grids_size))

    # generate a set of random numbers that do not repeat
    rand_set = gen_rand_set(batch_size * obj_count, alpha=0, beta=grids_size)

    # generate grids
    for i in range(batch_size):
        for j in range(obj_count):

            # generate a random number
            rand_grid_val = target_grid(bbox=torch.rand(4).numpy().tolist(),
                                        obj=torch.randint(0, OBJECT_CATEGORIES, (1,)).item())

            # yield a generated grid to the grids
            grids[i, :, rand_set[i * obj_count + j]] = rand_grid_val

    return grids


if __name__ == '__main__':
    grids = batch_grids_with_target_vals(batch_size=8, grids_size=64, obj_count=4)
    print(grids.shape)
    
    for i in range(8):
        for g in range(64):
            print(grids[i, :, g])
        print('\n')
