"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/10/24-18:23
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import cv2
import MNN
import time
import onnx
import torch
import cvzone
import onnxruntime
import torchvision
import numpy as np

from layers.functions.prior_box import PriorBox
# from utils.nms_wrapper import nms
from torchvision.ops import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode

#图像类别
classes_person=['__background__','face']

device = 'cpu' if torch.cuda.is_available() else 'cpu'

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def loadModel(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
def predict():
    x = torch.rand(size = (1,3,640,640)).to(device)
    pretrained_path = r'./weights/FaceBoxes.pth'
    model = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
    model = loadModel(model, pretrained_path)
    return x,model


def loadONNXObject(cnnModelPath = './weights/FaceBoxes.pth'):
    x,model = predict()
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            './onnx/faceboxes.onnx',  # 导出的ONNX名称
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['boxes','conf']  # 输出tensor的名称（名称自己决定）
        )


def InferenceTimeDetect(onnxPath = './onnx/faceboxes.onnx'):
    onnxSSDLite = onnxruntime.InferenceSession(onnxPath)
    video = "http://admin:admin@192.168.161.35:8081/video"
    # video = "http://192.168.161.35:8081/video"
    cap = cv2.VideoCapture(video)

    count = 0
    start_time = time.time()
    from data import cfg

    priorbox = PriorBox(cfg, image_size=(640, 640))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    print('prior data.size: {}'.format(prior_data.size()))

    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if ret == False:
            break
        frame = cv2.resize(frame, dsize=(640, 640))
        frame = cv2.flip(src=frame, flipCode=2)
        h, w = np.shape(frame)[:2]

        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        scale = scale.to(device)
        # 这里需要变换通道(H,W,C)=>(C,H,W)
        # 方式一：
        img -= (104, 117, 123)
        newImg = np.transpose(img, (2, 0, 1))
        # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        newImg = torch.Tensor(newImg)
        newImg = torch.unsqueeze(input=newImg, dim=0).numpy()
        # 计算开始时间
        start_time = time.time()

        loc, conf= onnxSSDLite.run(['boxes','conf'],{'input':newImg})
        loc, conf = torch.tensor(loc),torch.tensor(conf)

        # loc.size: torch.Size([1, 8525, 4])  conf.size: torch.Size([1, 8525, 2])
        print("loc.size: {}  conf.size: {}".format(loc.size(), conf.size()))

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.7)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores, dtype=torch.float32)

        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(boxes=boxes, scores=scores, iou_threshold=0.3)
        # do NMS
        boxes = boxes[keep]
        scores = scores[keep]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        # keep top-K faster NMS
        dets = dets[:750, :]
        for b in dets:
            if b[4] < 0.5:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255))

        # 计算结束时间
        end_time = time.time()
        FPS = round(1 / (end_time - start_time), 0)
        cv2.putText(img=frame, text='FPS: ' + str(FPS), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 0), thickness=2)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def InferenceMNN(mnnPath = 'MNNModel/squeezeNetFace.mnn',img_path = 'images/person_0.jpg'):
    interpreter = MNN.Interpreter(mnnPath)
    mnn_session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(mnn_session)
    imgTo = cv2.imread(img_path)
    imgTo = cv2.resize(imgTo, (320, 320)) / 255
    # 这里需要变换通道(H,W,C)=>(C,H,W)
    # 方式一：
    newImg = np.transpose(imgTo, (2, 0, 1))
    # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
    # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
    newImg = torch.Tensor(newImg)
    # 扩充维度，这里一定要注意，将其转化为numpy格式
    newImg = torch.unsqueeze(input=newImg, dim=0)
    input = MNN.Tensor((1,3,224,224),MNN.Halide_Type_Float,newImg,MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(input)
    interpreter.runSession(mnn_session)
    output_tensor = interpreter.getSessionOutput(mnn_session)
    print(type(output_tensor.getData()))
    print(output_tensor)


if __name__ == '__main__':
    from data import cfg
    priorbox = PriorBox(cfg, image_size=(640, 640))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    print('prior data.size: {}'.format(prior_data.size()))
    # loadONNXObject()
    InferenceTimeDetect()
    pass
