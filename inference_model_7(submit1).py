from glob import glob
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
model = YOLO("/home/jupyter/datasphere/project/runs/detect/train7/weights/last.pt") # model 7
lst = glob("/home/jupyter/datasphere/project/test_minprirodi_Parnokopitnie/*")
results = []
for i in lst:
    result = model([i])
    results.extend(result)
to_csv= []
non = []
i = 0
for result in results:
    if len(results[i].boxes.cls.cpu().tolist()) == 0:
        non.append(i)
        to_csv.append(1)
    else:
        to_csv.append(int(results[i].boxes.cls.cpu().tolist()[0]))
    i+=1

d = {'img_name':list(map(lambda x: x.split('/')[-1], lst)), 'class': to_csv}
df = pd.DataFrame(d)
df.to_csv("/home/jupyter/datasphere/project/sub.csv",index=False)