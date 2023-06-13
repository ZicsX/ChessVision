import random
import numpy as np
import onnxruntime as ort
import cv2
import time
import torch
import torchvision

class V5:
    def __init__(self, conf_thres=0.5, iou_thres=0.5, img_size=640):
        self.conf_thres=conf_thres
        self.iou_thres =iou_thres
        self.img_size=img_size
        self.name= ['black bishop', 'black king', 'black knight', 'black pawn', 'black queen', 'black rook', 'white bishop', 'white king', 'white knight', 'white pawn', 'white queen', 'white rook']

        # Initialize ONNXRuntime session   
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession('models/best.onnx', providers=self.providers)# self.cuda= cuda

    def __call__(self,image_or):
        #image preprocessing
        image, ratio, dwdh = self.letterbox(image_or)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        # print(im.shape)

        #onnxruntime session
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]:im}

        # ONNXRuntime inference
        outputs = self.session.run(outname, inp)[0]
        output= torch.from_numpy(outputs)
        out =self.non_max_suppression(output, self.conf_thres, self.iou_thres)[0]
        img=self.result(image_or,ratio, dwdh, out)
        return img


    def box_iou(self,box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
            prediction,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nm=0,  # number of masks
    ):
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output


    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def letterbox(self,img, img_size=640):
        height, width = img.shape[:2]
        ratio = min(img_size/width, img_size/height)
        new_height, new_width = int(height*ratio), int(width*ratio)
        img = cv2.resize(img, (new_width, new_height))
        top, bottom, left, right = 0, img_size-new_height, 0, img_size-new_width
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        return img, ratio, (top, left)

    # Display results
    def result(self,img,ratio, dwdh, out):
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.name)}   
        for i,(x0,y0,x1,y1,score,cls_id) in enumerate(out):
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = self.name[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(img,box[:2],box[2:],color,2)
            cv2.putText(img,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2) 
        return img
