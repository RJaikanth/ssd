import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils.helpers import LABEL_MAP_R, LABEL_CMAP


def draw_bboxes(image: np.array, bboxes: np.array, labels: np.array) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for (bbox, label) in zip(bboxes, labels):
        color = LABEL_CMAP[LABEL_MAP_R[label]]
        bbox = bbox * 300

        cv2.rectangle(
            image,
            tuple(bbox[:2]),
            tuple(bbox[2:]),
            color=color,
            thickness=2,
        )
        cv2.putText(
            image,
            LABEL_MAP_R[label],
            (bbox[0], int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            color,
            1,
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
