# Label map
LABELS = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
# Color map for bounding boxes of detected objects from
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 49),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (0, 0, 128),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 216, 177),
    (230, 190, 255),
    (128, 128, 128),
    (255, 255, 255),
]

LABEL_MAP = {k: v + 1 for v, k in enumerate(LABELS)}
LABEL_MAP["background"] = 0
LABEL_MAP_R = {v: k for k, v in LABEL_MAP.items()}
LABEL_CMAP = {k: COLORS[i] for i, k in enumerate(LABEL_MAP.keys())}
