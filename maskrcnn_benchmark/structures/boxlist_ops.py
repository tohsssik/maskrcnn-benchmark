# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    #[LY]
    #print('\ndebug at {}'.format(__file__))
    #_device = boxlist1.bbox.get_device()
    #_start_mem = torch.cuda.memory_allocated(device=_device)
    #print('before iou mem(MB): {}'.format(_start_mem/1024/1024)) 
    

    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)
    
    #[LY] try move to half here
    #area1 = boxlist1.area()
    #area2 = boxlist2.area()
    area1 = boxlist1.area().half()
    area2 = boxlist2.area().half()

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    #[LY] try move to half here
    #lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    #rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    lt = torch.max(box1[:, None, :2], box2[:, :2]).half()
    rb = torch.min(box1[:, None, 2:], box2[:, 2:]).half()
    #print(lt.shape,rb.shape)



    TO_REMOVE = 1

    #[LY] modify as FS (issue18) to reduce the memory
    #wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    #inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    #wh = rb.add_(lt.mul_(-1)).add_(TO_REMOVE).clamp_(min=0)
    
    wh = rb.sub_(lt).add_(TO_REMOVE).clamp_(min=0)
    inter = wh[:, :, 0].mul_(wh[:, :, 1])
    #[LY]
    #print('\ndebug at {}'.format(__file__))
    #print(type(area1),type(area2),type(inter))
    #print(area1.shape,area2.shape,inter.shape)
    #iou = inter / (area1[:, None] + area2 - inter)
    #iou = inter.div_(area1[:, None].add(area2).add_(inter.mul_(-1)))

    #[LY]
    #print('\ndebug at {}'.format(__file__))
    #_end_mem = torch.cuda.memory_allocated(device=_device)
    #print('after  iou mem(MB): {}'.format(_end_mem/1024/1024))
    ##print(lt.shape,rb.shape,wh.shape,inter.shape,iou.shape)
    ##print('inter and iou shape: {},{}'.format(inter.shape,iou.shape))
    #_target_tensor = inter
    #print('tensor.shape: {}'.format(_target_tensor.shape))
    #print('tensor.dtype: {}'.format(_target_tensor.dtype))
    #print('tensor.mem_size(MB): {}'.format(
    #    _target_tensor.element_size()*_target_tensor.nelement()/1024/1024
    #))
    #print('mem increased(MB): {}'.format((_end_mem-_start_mem)/1024/1024))


    iou = inter/(area1[:, None].add(area2).sub_(inter))
    #x1 = inter.mul_(-1)
    #x2 = area1[:,None].add(area2)
    #x3 = x2.add_(x1)
    #iou = inter.div_(x3)
    
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
