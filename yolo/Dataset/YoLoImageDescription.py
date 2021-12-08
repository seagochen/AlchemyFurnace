import yolo.Dataset.PascalImageDescription as PascalImageDescription

# define Pascal VOC categories
# Person: person
# Animal: bird, cat, cow, dog, horse, sheep
# Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
# Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
PIC_CATEGORIES = [
    'person',
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def object_name_to_index(obj_type: str):
    return PIC_CATEGORIES.index(obj_type)


def index_to_object_name(idx: int):
    return PIC_CATEGORIES[idx % len(PIC_CATEGORIES)]


def _bbox_to_yolo(ltx, lty, rbx, rby, grid_cells=7, width=448, height=448):
    """
    将绝对的 bounding box 转换为相对的 Grid 坐标

    :param ltx: bounding box 左上角顶点
    :param lty: bounding box 坐上顶点
    :param rbx: bounding box 右下脚顶点
    :param rby: bounding box 右下脚顶点
    :param grid_cells: 网格数
    :param width: 图片宽
    :param height: 图片高
    :return:
    """
    # 计算中心点相对位置，并归一化为 [0, 1] 区间
    cent_x = (rbx + ltx) / 2.
    cent_y = (rby + lty) / 2.

    grid_width = width / grid_cells
    cent_x = (cent_x % grid_width) / grid_width

    grid_height = height / grid_cells
    cent_y = (cent_y % grid_height) / grid_height

    # 计算Bounding box相对于图片的长宽，并对bounding box进行归一化
    bbox_w = rbx - ltx
    bbox_h = rby - lty
    bbox_w = bbox_w / width
    bbox_h = bbox_h / height

    return cent_x, cent_y, bbox_w, bbox_h


def _bbox_cent_idx(ltx, lty, rbx, rby, grid_cells=7, width=448, height=448):
    """
    计算 bounding box 所在的网格坐标

    :param ltx: bounding box 左上角顶点
    :param lty: bounding box 坐上顶点
    :param rbx: bounding box 右下脚顶点
    :param rby: bounding box 右下脚顶点
    :param grid_cells: 网格数
    :param width: 图片宽
    :param height: 图片高
    :return:
    """

    cent_x = (rbx + ltx) / 2.
    cent_y = (rby + lty) / 2.

    grid_width = width / grid_cells
    grid_height = height / grid_cells

    idx_i = idx_j = 0

    while cent_y > 0:
        cent_y = cent_y - grid_height
        idx_j += 1

    while cent_x > 0:
        cent_x = cent_x - grid_width
        idx_i += 1

    # 使用上面这种计算方法可以快速的找到中心点落入的GridCell坐标
    # 但是需要在计算结束后，i - 1, j - 1
    idx_i -= 1
    idx_j -= 1

    # 以 i, j的形式返回坐标，分别对应GridCell的横轴和纵轴坐标
    return idx_i, idx_j


class YoLoObjectDescription(object):

    def __init__(self, obj: PascalImageDescription.ObjectDescription, grid_cells=7, width=448, height=448):
        """
        YOLO 使用的坐标方式是相对坐标，及物体中心点对于GridCell的相对坐标，
        其目的在于把物体中心点坐标范围从[0, 447]转化为相对于GridCell中，
        并且归一化后使之落入[0, 1]之间。

        另外，YOLO的Grid是 7 x 7的，所以每个GridCell的大小是 (64, 64)

        :param obj:
        :param width
        :param height
        """

        """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
            并进行归一化"""

        # 对数据类型进行转换
        self.class_type = object_name_to_index(obj.type)

        # 物体是竖直的还是水平的
        self.pos = obj.pos

        # 记录图片的大小
        self.width = width
        self.height = height

        # 记录网格数，默认 7 x 7
        self.grid_cells = grid_cells

        # 图片是否被截取过
        self.truncated = obj.truncated
        self.truncated_val = int(obj.truncated_val)

        # 是否有物体遮盖
        self.occluded = obj.occluded
        self.occluded_val = int(obj.occluded_val)

        # 将 bounding box 的绝对坐标信息转换为 YOLO 需要的形式
        self.cent_x, self.cent_y, self.bbox_w, self.bbox_h = _bbox_to_yolo(
            ltx=obj.lt_x,
            lty=obj.lt_y,
            rbx=obj.rb_x,
            rby=obj.rb_y,
            grid_cells=grid_cells,
            width=width,
            height=height)

        # 得到 bounding box 的中心点所在的网格信息
        self.grid_i, self.grid_j = _bbox_cent_idx(
            ltx=obj.lt_x,
            lty=obj.lt_y,
            rbx=obj.rb_x,
            rby=obj.rb_y,
            grid_cells=grid_cells,
            width=width,
            height=height)

    def bbox(self):
        """
        返回bounding box绝对位置信息。
        （左上角X，左上角Y，BBox的绝对W，BBox的绝对Y）
        :return:
        """
        # 计算绝对长、宽
        width = int(self.width * self.bbox_w)
        height = int(self.height * self.bbox_h)

        # 计算绝对中心点位置
        cent_x = (self.cent_x + self.grid_i) / self.grid_cells * self.width
        cent_y = (self.cent_y + self.grid_j) / self.grid_cells * self.height

        # 计算左上角位置
        ltx = int(round(cent_x - width / 2.))
        lty = int(round(cent_y - height / 2.))

        return ltx, lty, width, height

    def grid_width(self):
        return int(self.width / self.grid_cells)

    def grid_height(self):
        return int(self.height / self.grid_cells)


class YoLoImageInfo(object):

    def __init__(self, annotation):
        self.filename = annotation['filename']
        self.width = int(annotation['size']['width'])
        self.height = int(annotation['size']['height'])
        self.depth = int(annotation['size']['depth'])
        self.objects = []

        self._parse_object(annotation)

    def __str__(self):
        val = {
            "filename": self.filename,
            "img_width": self.width,
            "img_height": self.height,
            "img_depth": self.depth,
            "objects:": self.objects}
        return str(val)

    def _parse_object(self, annotation):
        for obj in annotation['object']:
            img_obj = PascalImageDescription.ObjectDescription(obj)
            yolo_obj = YoLoObjectDescription(obj=img_obj, width=self.width, height=self.height)
            self.objects.append(yolo_obj)


# if __name__ == "__main__":
#     anno = {'name': 'bird', 'pose': 'vertical',
#             'bndbox': {'xmin': '90', 'ymin': '10', 'xmax': '130', 'ymax': '60'}, 'truncated': '0', 'occluded': '0'}
#
#     print(_bbox_cent_idx(ltx=70, lty=60, rbx=120, rby=100))
#     print(_bbox_to_yolo(ltx=70, lty=60, rbx=120, rby=100))
#
#     pascal_obj = PascalImageDescription.ObjectDescription(anno)
#     yolo_obj = YoLoObjectDescription(pascal_obj, width=448, height=448)
#     print(yolo_obj.bbox())
