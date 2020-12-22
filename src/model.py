from typing import Tuple

from math import sqrt
from itertools import product as product

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .utils.metrics import iou
from .utils.helpers import decimate
from .utils.bbox_ops import cxcy_to_xy, gcxgcy_to_cxcy, cxcy_to_gcxgcy, xy_to_cxcy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    """
    Base Model
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard Convolution Layersin VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = F.relu(self.conv6(out))
        conv7_feats = F.relu(self.conv7(out))

        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to conv layers and subsample to sizes of conv6 and conv7
        conv_fc6_weight = pretrained_state_dict["classifier.0.weight"].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict["classifier.0.bias"]
        state_dict["conv6.weight"] = decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict["conv6.bias"] = decimate(conv_fc6_bias, m=[4])

        conv_fc7_weight = pretrained_state_dict["classifier.3.weight"].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict["classifier.3.bias"]
        state_dict["conv7.weight"] = decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict["conv7.bias"] = decimate(conv_fc7_bias, m=[4])

        self.load_state_dict(state_dict)


class AuxiliaryConvolutions(nn.Module):
    """
    Auxiliary Convolutions
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out

        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        conv11_2_feats = out

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Prediction layers.
    """

    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4,
        }

        # Localization prediction convolutions (predict offsets w.r.t. prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes["conv11_2"] * 4, kernel_size=3, padding=1)

        # Class prediction
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes["conv11_2"] * n_classes, kernel_size=3, padding=1)

        # Initialize layers
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0)

    def forward(
        self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
    ):
        batch_size = conv4_3_feats.size(0)

        # Predict locatlozation boxes
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        # Predict Classification of objects in box.
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        cls_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)

        return locs, cls_scores


class SSD300(nn.Module):
    """
    SSD Module.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_conv = AuxiliaryConvolutions()
        self.pred_conv = PredictionConvolutions(n_classes)

        # We take L2 norm and rescale conv4_3_feats as it has considerably larger scales.
        # Initially set at 20, but is learned for each channel during backprop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)).to(device)
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        conv4_3_feats, conv7_feats = self.base(image)

        # Rescale conv4_3 after L2 normalization
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        # Run auxiliary convolutions
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_conv(conv7_feats)

        # Run prediction convolutions
        locs, classes_scores = self.pred_conv(
            conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
        )

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create default 8732 prior boxes as defined in the paper.

        Returns
        -------
        [torch.tensor]
            prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5, "conv10_2": 3, "conv11_2": 1}

        obj_scales = {
            "conv4_3": 0.1,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
            "conv11_2": 0.9,
        }

        aspect_ratios = {
            "conv4_3": [1.0, 2.0, 0.5],
            "conv7": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv8_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv9_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv10_2": [1.0, 2.0, 0.5],
            "conv11_2": [1.0, 2.0, 0.5],
        }

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            # These for loops iterates over every pixel location in the feature map
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append(
                            [cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)]
                        )

                        if ratio == 1:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError as e:
                                additional_scale = 1
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_boxes(
        self,
        predicted_locs: torch.tensor,
        predicted_scores: torch.tensor,
        min_score: float,
        max_overlap: float,
        top_k: int,
    ) -> Tuple[list, list, list]:
        """
        Decipher the 8732 locations and class scores (output of SSD300) to detect objects
        For each class perform NMS on boxes that are above a min threshold.

        Parameters
        ----------
        predicted_locs : torch.tensor
            predicted locations w.r.t the 8732 prior boxes, (N, 8732, 4)
        predicted_scores : torch.tensor
            class scores for each of the encoded location, (N, 8732, 4)
        min_score : float
            Minimum threshold for a box to be considered a match for a certain class
        max_overlap : float
            maximum overlap two boxes can have so that the one with the lower score is not suppressed
        top_k : int
            if there a lot of resulting detection across all classes, keep only the top 'k'


        Returns
        -------
        Tuple[list, list, list]:
            Detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labesl and scores for all images
        all_image_boxes, all_image_labels, all_image_scores = [], [], []
        assert n_priors == predicted_scores.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates ffrom the form we regressed predicted boxes,
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4)

            # Lists to score boxes and scores for this image.
            image_boxes, image_labels, image_scores = [], [], []

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                # Sort predictedboxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                overlap = iou(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # NMS
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue
                    suppress = torch.max(suppress.overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c])).to(device)
                image_boxes.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for background
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0.0, 0.0, 1.0, 1.0]])).to(device)
                image_labels.append(torch.FloatTensor([0])).to(device)
                image_scores.append(torch.FloatTensor([0.0])).to(device)

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_boxes[sort_ind][:top_k]

            # Append to lists that store all predicted boxes and scores
            all_image_boxes.append(image_boxes)
            all_image_labels.append(image_labels)
            all_image_scores.append(image_scores)

        return all_image_boxes, all_image_labels, all_image_scores  # Lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox Loss, a function for object detection.

    This is a combination of:
        1. Localization loss for the predicted locations of the boxes.
        2. Confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1):
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = iou(boxes[i], self.priors_xy)

            # For each prior, find the object that has maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            # We don't want a situation where an object is not represented in out positive priors-
            # An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # All priors with the object may be assigned as background based on the threshold

            # To remedy this-
            # First find the prior that has maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)

            # Then, assign each object to the corresponding maximum-overlap-prior
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5
            overlap_for_each_prior[prior_for_each_object] = 1

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]
            # Set priors whose overloaps with objects are less than the threshold to be background
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        # Identify priors that are positive
        positive_priors = true_classes != 0

        # LOCALIZATION LOSS
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)

        # We  already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]

        # Find out which priors are hard-negative
        # sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hardness_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hardness_negatives]

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        # TOTAL LOSS

        # print(conf_loss, loc_loss)

        return conf_loss + self.alpha * loc_loss
