import torch


def find_intersection(set_1: torch.TensorType, set_2: torch.TensorType) -> torch.Tensor:
    """
    Find the intersection area of 2 sets of bounding boxes.

    Parameters
    ----------
    set_1 : torch.TensorType
        A Tensor of dimensions (n1, 4)
    set_2 : torch.TensorType
        A Tensor of dimensions (n2, 4)

    Returns
    -------
    torch.Tensor
        A Tensor of dimensions (n1, n2) containing intersction of every box combination in set_1 and set_2
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2 ,2)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def iou(set_1: torch.TensorType, set_2: torch.TensorType) -> torch.Tensor:
    """
    Find the Intersection Over Union of 2 sets of boxes.

    Parameters
    ----------
    set_1 : torch.TensorType
        First set of bounding boxes
    set_2 : torch.TensorType
        Second set of bounding boxes

    Returns
    -------
    torch.Tensor
        IoU of every box in Set1 with every box in Set2
    """
    # Find Intersection
    intersection = find_intersection(set_1, set_2)

    # Find areas of each box in both sets.
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union  # (n1, n2)
