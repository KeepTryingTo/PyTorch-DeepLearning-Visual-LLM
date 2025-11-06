"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/10/24-16:49
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
from PIL import Image
from predict_faceDetect import drawRectangle

#图像类别
classes_person=['__background__','person']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadModel():
    model = torchvision.models.mobilenet_v3_small(pretrained = True,progress = True)
    model = model.eval().to(device)
    return model

def predict():
    x = torch.rand(size = (1,3,224,224)).to(device)
    model = loadModel()
    predictions = model(x)
    print(predictions.shape)

    return x,model

def torch2ONNX():
    x,model = predict()
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            'MobileNetV3.onnx',    #导出的ONNX名称
            opset_version=11,      #ONNX算子集版本
            input_names=['input'], #输入Tensor的名称（名称自己决定）
            output_names=['predictions']#输出tensor的名称（名称自己决定）
        )
    return model

def loadONNX(onnxPath='onnxModel/MobileNetV3.onnx'):
    #加载ONNX模型
    modelONNX = onnx.load(onnxPath)
    #检查模型格式是否正确
    onnx.checker.check_model(modelONNX)
    #打印ONNX的计算图
    # print(onnx.helper.printable_graph(modelONNX.graph))
    return modelONNX

def loadONNXObject(cnnModelPath = 'CNNModel/myModelBestssd320_2.pth'):
    onnxModel = torch.load(cnnModelPath, map_location=torch.device('cpu'))
    onnxModel = onnxModel.eval().to(device)
    x = torch.rand(size = (1,3,320,320)).to(device)
    with torch.no_grad():
        torch.onnx.export(
            onnxModel,
            x,
            'onnxModel/SSDLite320.onnx',  # 导出的ONNX名称
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['boxes','scores','labels']  # 输出tensor的名称（名称自己决定）
        )
    loadONNX(onnxPath='onnxModel/SSDLite320.onnx')


def loadONNXClassify(cnnModelPath = 'CNNModel/squeezeNetFace.pth'):
    onnxModel = torch.load(cnnModelPath, map_location=torch.device('cpu'))
    onnxModel = onnxModel.eval().to(device)
    x = torch.rand(size = (1,3,224,224)).to(device)
    with torch.no_grad():
        torch.onnx.export(
            onnxModel,
            x,
            'onnxModel/squeezeNetFace.onnx',  # 导出的ONNX名称
            opset_version=11,  # ONNX算子集版本
            input_names=['input'],  # 输入Tensor的名称（名称自己决定）
            output_names=['predictions']  # 输出tensor的名称（名称自己决定）
        )
    loadONNX(onnxPath='onnxModel/squeezeNetFace.onnx')

def InferenceSignalImage(onnxPath = 'onnxModel/SSDLite320.onnx',img_path = 'images/person_0.jpg'):
    onnxSSDLite = onnxruntime.InferenceSession(onnxPath)
    x = torch.randn(size = (1,3,320,320)).numpy()
    input = {'input':x}
    output = onnxSSDLite.run(['boxes','scores','labels'],input)[0]
    print('output.shape: {}'.format(output.shape))
    # print('output: {}'.format(output))

    imgTo = cv2.imread(img_path)
    imgTo = cv2.resize(imgTo, (320, 320)) / 255
    # 这里需要变换通道(H,W,C)=>(C,H,W)
    # 方式一：
    newImg = np.transpose(imgTo, (2, 0, 1))
    # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
    # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
    newImg = torch.Tensor(newImg)
    # 扩充维度，这里一定要注意，将其转化为numpy格式
    newImg = torch.unsqueeze(input=newImg, dim=0).numpy()

    input = {'input':newImg}
    boxes,scores,labels = onnxSSDLite.run(['boxes','scores','labels'],input)

    print('boxes.shape: {}'.format(np.shape(boxes)))
    print('boxes: {}'.format(torch.tensor(boxes)))

    print('labels.shape: {}'.format(np.shape(labels)))
    print('labels: {}'.format(torch.tensor(labels)))

    print('scores.shape: {}'.format(np.shape(scores)))
    print('scores: {}'.format(torch.tensor(scores)))

    drawRectangle(boxes=boxes,labels=labels,scores=scores,img_path = img_path)

def InferenceTimeDetect(onnxPath = 'onnxModel/SSDLite320.onnx'):
    onnxSSDLite = onnxruntime.InferenceSession(onnxPath)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 320))
        newImg = frame / 255

        frame_ = cv2.flip(src=frame, flipCode=2)
        size = frame.shape
        # 这里需要变换通道(H,W,C)=>(C,H,W)
        # 方式一：
        newImg = np.transpose(newImg, (2, 0, 1))
        # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        newImg = torch.Tensor(newImg)
        newImg = torch.unsqueeze(input=newImg, dim=0).numpy()
        # 计算开始时间
        start_time = time.time()

        boxes,scores,labels = onnxSSDLite.run(['boxes','scores','labels'],{'input':newImg})
        boxes,labels,scores = torch.tensor(boxes),torch.tensor(labels),torch.tensor(scores)

        for k in range(len(labels)):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0])
            yleft = int(boxes[k][1])
            xright = int(boxes[k][2])
            yright = int(boxes[k][3])

            class_id = labels[k].item()

            confidence = scores[k].item()
            # 这里只输出检测是人并且概率值最大的
            if class_id == 1 and confidence > 0.8:
                text = classes_person[class_id] + ': ' + str('{:.4f}'.format(confidence))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
                break

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
    # torch2ONNX()
    # loadONNX()
    # loadONNXObject()
    # loadONNXClassify()
    # InferenceSignalImage()
    InferenceTimeDetect()
    # InferenceMNN()
    pass
