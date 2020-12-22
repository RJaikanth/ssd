import torch


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding boxes predicted by the model, since they are encoded.

    Inverse of encoded function

    Parameters
    ----------
    gcxgcy : torch.tensor
        Encoded Bounding boxes (n_priors, 4)
    priors_cxcy : torch.tensor
        Prior boxes with respect to which the encoding must be performed (n_priors, 4)

    Returns
    -------
    torch.tensor
        Decoded bounding boxes in center-size form (n_priors, 4)
    """
    return torch.cat(
        [
            gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # cx, cy
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:],  # w, h
        ],
        1,
    )


def cxcy_to_xy(cxcy: torch.tensor) -> torch.tensor:
    """
    Convert boundingg boxes from boundary coordinate (xmin, ymin, xmax, ymax) to center-size coordinates.

    Parameters
    ----------
    xy : torch.Tensor
        Bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)

    Returns
    -------
    [torch.tensor]
        Bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat(
        [cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1  # x_min, y_min
    )  # x_max, y_max


def cxcy_to_gcxgcy(cxcy: torch.tensor, priors_cxcy: torch.tensor) -> torch.tensor:
    """
    Encode bounding boxes (that are in center-size form) w.r.t to the corresponding prior boxes (that are in the center-size form)

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of prior box, and convert to the log-space.

    Parameters
    ----------
    cxcy : torch.tensor
        Bounding boxes in center-size coordinates (n_priors, 4)
    priors_cxcy : torch.tensor
        Prior boxes with respect to which the encoding must be performed (n_priors, 4)

    Returns
    -------
    torch.tensor
        Encoded bounding boxes in center-size form (n_priors, 4)
    """
    return torch.cat(
        [
            (cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
            torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5,
        ],
        1,
    )  # g_w, g_h


def xy_to_cxcy(xy: torch.tensor) -> torch.tensor:
    """
    Convert boundingg boxes from boundary coordinate (xmin, ymin, xmax, ymax) to center-size coordinates.

    Parameters
    ----------
    xy : torch.Tensor
        Bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)

    Returns
    -------
    [torch.tensor]
        Bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)

    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)
