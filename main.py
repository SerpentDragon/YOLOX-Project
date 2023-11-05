import os
import cv2
import torch
import argparse
from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str, nargs=1, required=True, help="Specify name of your model")
parser.add_argument("--ckpt", "-c", type=str, nargs=1, required=True, help="Specify path to your model")
parser.add_argument("--path", "-p", type=str, nargs=1, required=True, help="Specify path to your input video")
parser.add_argument("--conf", type=float, nargs=1, required=False, help="Specify confidence threshold")
args = parser.parse_args()

exp = get_exp(exp_name=args.name[0])
model = exp.get_model()
model.eval()

confidence = args.conf[0] if args.conf is not None else 0


ckpt = torch.load(args.ckpt[0], map_location="cpu")
model.load_state_dict(ckpt["model"])

if os.path.exists(args.path[0]):
    inputVideo = cv2.VideoCapture(args.path[0])

    frame_width = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(inputVideo.get(cv2.CAP_PROP_FPS))
    frame_duration = 1 / frame_rate
    fourcc = cv2.VideoWriter.fourcc('X', 'V', 'I', 'D')
else:
    print("Video file not found!")
    exit()

result_path = "YOLOXProject_outputs"
if not os.path.exists(result_path):
    os.mkdir(result_path)

outputPath = os.path.splitext(args.path[0])
outputVideo = cv2.VideoWriter(os.path.join(result_path, os.path.basename(outputPath[0] + "_output" + outputPath[1])),
                              fourcc, frame_rate, (frame_width, frame_height))

counter = 0

while True:
    ret, frame = inputVideo.read()
    if not ret:
        break

    image, r = preproc(frame, (640, 640))
    image = torch.from_numpy(image).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(image)
        outputs = postprocess(outputs, exp.num_classes, 0.1, exp.nmsthre, class_agnostic=True)

        if outputs[0] is not None:
            bboxes = outputs[0][:, 0:4] / r
            cls = outputs[0][:, 6]
            scores = outputs[0][:, 4] * outputs[0][:, 5]

            vis_res = vis(frame, bboxes, scores, cls, confidence, COCO_CLASSES)
        else:
            vis_res = frame

        outputVideo.write(vis_res)

    counter += 1

inputVideo.release()
outputVideo.release()
