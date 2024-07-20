import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/final/lscd/runs/train/yolov8-LSCD/weights/best.pt')
    model.val(data='/final/datasets/dataset_visdrone/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8-LSCD',
              )