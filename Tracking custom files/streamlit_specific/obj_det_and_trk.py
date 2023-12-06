import os
import sys
import cv2
import time
import torch
import execjs
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

#---------------Object Tracking---------------
import skimage
from sort import *


#-----------Object Blurring-------------------
blurratio = 40


#.................. Tracker Functions .................
"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

""" Converts hex values to rgb values. """
def hex_to_rgb(hex_string):
    # Remove the hash sign if present
    hex_string = hex_string.lstrip('#')
    # Convert pairs of hex digits to decimal numbers and create a tuple
    return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
js_function = """
function pickTextColorBasedOnBgColorAdvanced(bgColor, lightColor, darkColor) {
    var color = (bgColor.charAt(0) === '#') ? bgColor.substring(1, 7) : bgColor;
    var r = parseInt(color.substring(0, 2), 16); // hexToR
    var g = parseInt(color.substring(2, 4), 16); // hexToG
    var b = parseInt(color.substring(4, 6), 16); // hexToB
    var uicolors = [r / 255, g / 255, b / 255];
    var c = uicolors.map((col) => {
        if (col <= 0.03928) {
            return col / 12.92;
        }
        return Math.pow((col + 0.055) / 1.055, 2.4);
    });
    var L = (0.2126 * c[0]) + (0.7152 * c[1]) + (0.0722 * c[2]);
    return (L > 0.179) ? darkColor : lightColor;
}
"""

# Compile the JavaScript function using PyExecJS
js = execjs.compile(js_function)

# Define the light color and dark color in hex format
light_color = '#FFFFFF' # white
dark_color = '#000000' # black

def get_text_color(bg_color):
    # Convert the background rgb values to hex format
    bg_color_hex = '#%02x%02x%02x' % bg_color
    # Call the function to get the text color in hex format
    text_color_hex = js.call('pickTextColorBasedOnBgColorAdvanced',bg_color_hex, light_color, dark_color)
    text_color = hex_to_rgb(text_color_hex)
    return text_color


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), dets_to_sort=None, color_bee=(255,255,0), color_hornet=(255,0,0), text_bee=None, text_hornet=None):
    bees = 0
    hornets = 0
    for i, box in enumerate(bbox):
        if names[int(categories[i])] == 'Bees':
          bees += 1
        else:
          hornets += 1
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        conf = dets_to_sort[i, 4] if dets_to_sort is not None else 0 # get the confidence score from the dets_to_sort array
        label = f'{id}: {names[int(categories[i])]} {conf:.2f}' # use the conf variable for the label

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2),color_bee if names[int(categories[i])] == 'Bees' else color_hornet, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color_bee if names[int(categories[i])] == 'Bees' else color_hornet, -1)
        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_bee if names[int(categories[i])] == 'Bees' else text_hornet, 2)            
        cv2.circle(img, data, 3, color_bee if names[int(categories[i])] == 'Bees' else color_hornet,-1)
        cv2.putText(img, f'Hornets: {hornets}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color_hornet, 2) 
        cv2.putText(img, f'Bees: {bees}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color_bee, 2) 
    return img
#..............................................................................


@torch.no_grad()
def detect(weights=ROOT / 'yolov5n.pt',
        source=ROOT / 'yolov5/data/images', 
        data=ROOT / 'yolov5/data/coco128.yaml',  
        imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,  
        max_det=1000, device='cpu',  view_img=False,  
        save_txt=False, save_conf=False, save_crop=False, 
        nosave=False, classes=None,  agnostic_nms=False,  
        augment=False, visualize=False,  update=False,  
        project=ROOT / 'runs/detect',  name='exp',  
        exist_ok=False, line_thickness=2,hide_labels=False,  
        hide_conf=False,dnn=False,display_labels=False,
        blur_obj=False, color_bee=(255,255,0), color_hornet=(255,0,0)):
    
    save_img = not nosave and not source.endswith('.txt') 

    # if '#' in color_bee:
    #     # Convert the hex value to rgb using the function
    #     color_bee = hex_to_rgb(color_bee)
    text_bee = get_text_color(color_bee)

    # # Check if the value of hornet_color contains a #
    # if '#' in color_hornet:
    #     # Convert the hex value to rgb using the function
    #     color_hornet = hex_to_rgb(color_hornet)
    text_hornet = get_text_color(color_hornet)
    #.... Initialize SORT .... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
    track_color_id = 0
    #......................... 
    
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    set_logging()
    device = select_device(device)
    # half &= device.type != 'cpu'  

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  

    # half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
    # if pt or jit:
    #     model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1 
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    t0 = time.time()
    
    dt, seen = [0.0, 0.0, 0.0], 0
    
    for path, im, im0s, vid_cap, s in dataset:
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Clip boxes to image boundaries
                det[:, :4] = det[:, :4].clamp(0, im0.shape[0])
                # Convert class indices to positive integers
                det[:, -1] = det[:, -1].abs().int()
                # Clip confidence scores to 0-1 range
                det[:, -2] = det[:, -2].clamp(0, 1)
                # Filter detections by confidence threshold
                det = det[det[:, -2] > conf_thres]
                # Create an empty array for dets_to_sort
                dets_to_sort = np.empty((0, 6))
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    # Append the input detections to dets_to_sort
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                 
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()
                

                #loop over tracks
                for track in tracks:
                    [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                            (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                            (153, 251, 152), thickness=line_thickness) for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 
                
                if len(tracked_dets)>0:
                  bbox_xyxy = tracked_dets[:,:4]
                  identities = tracked_dets[:, 8]
                  categories = tracked_dets[:, 4]
                  draw_boxes(im0, bbox_xyxy, identities, categories, names, dets_to_sort=dets_to_sort, color_bee=color_bee, color_hornet=color_hornet, text_bee=text_bee, text_hornet=text_hornet) # pass the confidence score to the draw_boxes function

                    

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) 
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        print("Frame Processing!")
    print("Video Exported Success")

    if update:
        strip_optimizer(weights)
    
    if vid_cap:
        vid_cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    parser.add_argument('--color-bee', default='#FFFF00', type=str, help='Choose color for bee boxes')
    parser.add_argument('--color-hornet', default='#FF0000', help='Choose color for hornet boxes')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)