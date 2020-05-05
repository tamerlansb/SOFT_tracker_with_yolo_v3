import numpy as np
import cv2
import torch


def postprocessing(detections, num_classes, obj_conf_thr=0.5, nms_thr=0.4):
    # Zero bounding box with objectioness confidence score less than threshold
    obj_conf_filter = (detections[:, :, 4] > obj_conf_thr).float().unsqueeze(2)
    detections = detections * obj_conf_filter

    # Transform bounding box coordinates to two corners
    box = detections.new(detections[:, :, :4].shape)
    box[:, :, 0] = detections[:, :, 0] - detections[:, :, 2] / 2
    box[:, :, 1] = detections[:, :, 1] - detections[:, :, 3] / 2
    box[:, :, 2] = box[:, :, 0] + detections[:, :, 2]
    box[:, :, 3] = box[:, :, 1] + detections[:, :, 3]
    detections[:, :, :4] = box

    num_batches = detections.shape[0]
    # results = torch.Tensor().to(device)
    results = list()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for b in range(num_batches):
        batch_results = torch.Tensor().to(device)
        img_det = detections[b]

        max_class_score, max_class_idx = torch.max(img_det[:, 5:5 + num_classes], 1)
        img_det = torch.cat((img_det[:, :5],
                             max_class_score.float().unsqueeze(1),
                             max_class_idx.float().unsqueeze(1)
                             ), 1)
        # img det - [b1_x, b1_y, b2_x, b2_y, obj_conf, class_score, class]

        # Remove zeroed rows
        nonzero_idx = img_det[:, 4].nonzero()
        img_det = img_det[nonzero_idx, :].view(-1, 7)

        if img_det.shape[0] == 0:
            results.append(batch_results.cpu())
        else:
            # Get the classes
            img_classes = torch_unique(img_det[:, -1])
            for c in img_classes:
                # Select rows with "c" class and sort by the class score
                class_img_det = img_det[(img_det[:, -1] == c).nonzero().squeeze()]
                # If there is only one detection, it will return a 1D tensor. Therefore, we perform a view to keep it in 2D
                class_img_det = class_img_det.view(-1, 7)
                # Sort by objectness score
                _, sort_idx = class_img_det[:, 4].sort(descending=True)
                class_img_det = class_img_det[sort_idx]

                iou = iou_vectorized(class_img_det)
                # Alert: There's another loop operation in nms function
                class_img_det = nms(class_img_det, iou, nms_thr)
                batch_results = torch.cat((batch_results, class_img_det), 0)

            results.append(batch_results.cpu())

    return results


def torch_unique(inp, CUDA=True):
    if CUDA:
        inp_cpu = inp.detach().cpu()

    res_cpu = torch.unique(inp_cpu)
    res = inp.new(res_cpu.shape)
    res.copy_(res_cpu)

    return res


def iou_vectorized(bbox):
    num_box = bbox.shape[0]

    bbox_leftTop_x = bbox[:, 0]
    bbox_leftTop_y = bbox[:, 1]
    bbox_rightBottom_x = bbox[:, 2]
    bbox_rightBottom_y = bbox[:, 3]

    inter_leftTop_x = torch.max(bbox_leftTop_x.unsqueeze(1).repeat(1, num_box), bbox_leftTop_x)
    inter_leftTop_y = torch.max(bbox_leftTop_y.unsqueeze(1).repeat(1, num_box), bbox_leftTop_y)
    inter_rightBottom_x = torch.min(bbox_rightBottom_x.unsqueeze(1).repeat(1, num_box), bbox_rightBottom_x)
    inter_rightBottom_y = torch.min(bbox_rightBottom_y.unsqueeze(1).repeat(1, num_box), bbox_rightBottom_y)

    inter_area = torch.clamp(inter_rightBottom_x - inter_leftTop_x, min=0) * torch.clamp(
        inter_rightBottom_y - inter_leftTop_y, min=0)
    bbox_area = (bbox_rightBottom_x - bbox_leftTop_x) * (bbox_rightBottom_y - bbox_leftTop_y)
    union_area = bbox_area.expand(num_box, -1) + bbox_area.expand(num_box, -1).transpose(0, 1) - inter_area

    iou = inter_area / union_area
    return iou


# Iterate through the bounding boxes and remove rows accordingly
def reduce_row_by_column(inp):
    i = 0
    while i < inp.shape[0]:
        remove_row_idx = inp[i][1].item()
        if inp[i][0] != remove_row_idx and i < inp.shape[0]:
            keep_mask = (inp[:, 0] != remove_row_idx).nonzero().squeeze()
            inp = inp[keep_mask]
        i += 1
    return inp


# bbox is expected to be sorted by class score in descending order
def nms(bbox, iou, nms_thres):
    # Create a mapping that indicates which row has iou > threshold
    remove_map = (iou > nms_thres).nonzero()
    remove_map = reduce_row_by_column(remove_map)

    remove_idx = torch_unique(remove_map[:, 0])
    res_bbox = bbox[remove_idx]

    return res_bbox