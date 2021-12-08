class ObjectDescription(object):

    def __init__(self, description: dict):
        self.type = description['name']
        self.pos = description['pose']
        # left top coordinates
        self.lt_x = int(description['bndbox']['xmin'])
        self.lt_y = int(description['bndbox']['ymin'])

        # right bottom coordinates
        self.rb_x = int(description['bndbox']['xmax'])
        self.rb_y = int(description['bndbox']['ymax'])

        if description['truncated'] == "0":
            self.truncated = False
            self.truncated_val = "0"
        else:
            self.truncated = True
            self.truncated_val = description['truncated']

        if description["occluded"] == "0":
            self.occluded = False
            self.occluded_val = "0"
        else:
            self.occluded = True
            self.occluded_val = description["occluded"]

    def __str__(self):
        val = {
            "type": self.type,
            "pos": self.pos,
            "truncated": self.truncated_val,
            "occluded": self.occluded_val,
            "bndbox": [self.lt_x, self.lt_y, self.rb_x, self.rb_y]}
        return str(val)

    def bbox(self):
        """
        bounding box
        :return:
        """
        return self.lt_x, self.lt_y, self.rb_x, self.rb_y


# define an image information class
class PascalImageInfo(object):

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
            img_obj = ObjectDescription(obj)
            self.objects.append(img_obj)
