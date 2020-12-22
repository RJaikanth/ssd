import os
import json

from .utils.helpers import parse_annotation
from .utils.defaults import LABEL_MAP


def create_data_lists(config: dict) -> None:
    """
    Create Data lists for containg training, validation and test sets.

    Parameters
    ----------
    config: dict
        Config used for creating data lists.

    Returns
    -------
    None:
        Saves 5 JSON files at the output folder containing path train and test images with the respective objects

    """
    assert "voc07" in config.keys(), "Please add the voc07 path in the config"
    assert "voc12" in config.keys(), "Please add the voc12 path in the config"
    assert "lists_folder" in config.keys(), "Please add the lists_folder path in the config"

    # Get paths
    voc07path = os.path.abspath(config["voc07"])
    voc12path = os.path.abspath(config["voc12"])
    lists_folder = os.path.abspath(config["lists_folder"])

    train_images, train_objects = [], []
    num_objects = 0

    # Training Data
    for path in [voc07path, voc12path]:
        # Find IDs of images in training data.
        with open(os.path.join(path, "ImageSets/Main/trainval.txt")) as f:
            ids = f.read().splitlines()

        for id_ in ids:
            # Parse annotation XML File
            objects = parse_annotation(os.path.join(path, "Annotations", id_ + ".xml"))
            if len(objects["boxes"]) == 0:
                continue
            num_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, "JPEGImages", id_ + ".jpg"))
    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(lists_folder, "TRAIN_images.json"), "w") as f:
        json.dump(train_images, f)
        f.close()
    with open(os.path.join(lists_folder, "TRAIN_objects.json"), "w") as f:
        json.dump(train_objects, f)
        f.close()
    with open(os.path.join(lists_folder, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f)
        f.close()

    # Testing Data
    test_images, test_objects = [], []
    num_objects = 0

    with open(os.path.join(voc07path, "ImageSets/Main/test.txt")) as f:
        ids = f.read().splitlines()

    for id_ in ids:
        # Parse annotation XML File
        objects = parse_annotation(os.path.join(voc07path, "Annotations", id_ + ".xml"))
        if len(objects["boxes"]) == 0:
            continue
        num_objects += len(objects)
        test_objects.append(objects)
        test_images.append(os.path.join(path, "JPEGImages", id_ + ".jpg"))
    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(lists_folder, "TEST_images.json"), "w") as f:
        json.dump(test_images, f)
        f.close()
    with open(os.path.join(lists_folder, "TEST_objects.json"), "w") as f:
        json.dump(test_objects, f)
        f.close()

    print(
        "There are {} train images containig a total of {} objects.".format(
            len(train_images), len(train_objects)
        )
    )
    print(
        "There are {} test images containig a total of {} objects.".format(
            len(test_images), len(test_objects)
        )
    )
    print("Files have been saved to {} as json lists".format(lists_folder))
