import torchvision.datasets as datasets

DATA_PATH = "../../Data/PASCAL_VOC"

IMAGE_SET = ["train",       # image set with 20 object categories, with 5717 objects
             "trainval",    # image set with 20 object categories, with 11540 objects
             "val"]         # image set with 20 object categories, with 5823 objects

#YEARS = ["2007", "2009", "2010", "2011", "2012"]
YEARS = ["2012"]


if __name__ == "__main__":
    for year in YEARS:
        for set in IMAGE_SET:
            datasets.VOCDetection(DATA_PATH, year=year, image_set=set, download=True)

