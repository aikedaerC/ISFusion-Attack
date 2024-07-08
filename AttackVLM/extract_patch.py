import cv2
import os
from tqdm import tqdm
import numpy as np
import json
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors




class_mapping = {
    'person': 'persons',
    'bicycle': 'motorcycles',
    'car': 'cars',
    'motorcycle': 'motorcycles',
    'airplane': "UN",
    'bus': 'cars',
    'train': 'cars',
    'truck': 'cars',
    'boat': "UN",
    'traffic light': 'traffic lights',
    'fire hydrant': "UN",
    'stop sign': "road signals",
    'parking meter': "UN",
    'bench': "UN",
    'bird': "UN",
    'cat': "UN",
    'dog': "UN",
    'horse': "UN",
    'sheep': "UN",
    'cow': "UN",
    'elephant': "UN",
    'bear': "UN",
    'zebra': "UN",
    'giraffe': "UN",
    'backpack': "UN",
    'umbrella': "UN",
    'handbag': "UN",
    'tie': "UN",
    'suitcase': "UN",
    'frisbee': "UN",
    'skis': "UN",
    'snowboard': "UN",
    'sports ball': "UN",
    'kite': "UN",
    'baseball bat': "UN",
    'baseball glove': "UN",
    'skateboard': "UN",
    'surfboard': "UN",
    'tennis racket': "UN",
    'bottle': "UN",
    'wine glass': "UN",
    'cup': "UN",
    'fork': "UN",
    'knife': "UN",
    'spoon': "UN",
    'bowl': "UN",
    'banana': "UN",
    'apple': "UN",
    'sandwich': "UN",
    'orange': "UN",
    'broccoli': "UN",
    'carrot': "UN",
    'hot dog': "UN",
    'pizza': "UN",
    'donut': "UN",
    'cake': "UN",
    'chair': "UN",
    'couch': "UN",
    'potted plant': "UN",
    'bed': "UN",
    'dining table': "UN",
    'toilet': "UN",
    'tv': "UN",
    'laptop': "UN",
    'mouse': "UN",
    'remote': "UN",
    'keyboard': "UN",
    'cell phone': "UN",
    'microwave': "UN",
    'oven': "UN",
    'toaster': "UN",
    'sink': "UN",
    'refrigerator': "UN",
    'book': "UN",
    'clock': "UN",
    'vase': "UN",
    'scissors': "UN",
    'teddy bear': "UN",
    'hair drier': "UN",
    'toothbrush': "UN"
}


def init_patch_square(num_patch):
    patches = []
    patch = np.random.rand(1, 3, 224, 224) # [bt, ch, h, w] over [0,1]
    for i in range(num_patch):
        patches.append(patch)
        
    return patches, patch.shape

def process_one_img_det(im0):
    model = YOLO("yolov8n.pt")
    names = model.names

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    # annotator = Annotator(im0, line_width=2, example=names)

    if len(boxes)==0:
        return False
    
    for box, cls in zip(boxes, clss):
        obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] # obj.shape like (139, 64, 3)
        real_label = names[int(cls)]
        if class_mapping[real_label] in ["cars", "motorcycles", "traffic lights", "road signals", "persons"]:
            return True
    return False

def patch_transform(patch, data_shape, patch_shape, image_size, num_patch, im0, filename, reg_book, output_dir):
    """
    patch: list of patch, [(1,3,81,81), ]
    image_size: size of image, h and w , [h,w]
    """
    row_num = data_shape[3]//512
    column_num = data_shape[2]//512
    count = 0
    for j in range(row_num):
        for m in range(column_num):
            x = np.zeros(data_shape) # shape is (1, 3, realh,realw)
        
            # get shape
            m_size = patch_shape[-1] # (1, 3, 224, 224)
            for i in range(x.shape[0]): # for batch size

                # random rotation
                # rot = np.random.choice(4)
                # for k in range(patch[j*row_num+m+1][i].shape[0]): # channel
                #     patch[j*row_num+m+1][i][k] = np.rot90(patch[j*row_num+m+1][i][k], rot)
                
                # random location
                random_x = 4 + j * m_size #np.random.choice(image_size[0])
                if random_x + m_size > x.shape[-2]:
                    while random_x + m_size > x.shape[-2]:
                        random_x = np.random.choice(image_size[0])
                random_y = 4 + m * m_size #np.random.choice(image_size[1])
                if random_y + m_size > x.shape[-1]:
                    while random_y + m_size > x.shape[-1]:
                        random_y = np.random.choice(image_size[1])
        
            real_label = "background"
            box = [random_x, random_y, random_x+patch_shape[-1], random_y+patch_shape[-1]]
            obj = im0[box[1]:box[3], box[0]:box[2]] # obj.shape like (139, 64, 3)

            exist_obj = process_one_img_det(obj)
            if exist_obj:
                reg_book[filename]["exist"].append("YES")
            else:
                reg_book[filename]["exist"].append("NO")

            reg_book[filename]["bbox"].append(box)
            out_path = os.path.join(output_dir, real_label)
            os.makedirs(out_path, exist_ok=True)
            rand_path = os.path.join(out_path, filename[:-4][-2:] + "_" + str(count)+".jpg")
            reg_book[filename]["obj"].append(rand_path)
            cv2.imwrite(rand_path, obj)
            count += 1


def extra_patch(image_dir, output_dir, reg_book_path):
    NUM_PATCH = 24 # at most: 32
    patch, patch_shape = init_patch_square(num_patch=NUM_PATCH) 

    reg_book = {} # {"000030.jpg":{"bbox":[], "obj":[]},}

    for filename in tqdm(os.listdir(image_dir)):
        reg_book[filename] = {"bbox":[], "obj":[], "exist": []}
        # Read the image
        img_path = os.path.join(image_dir, filename)
        im0 = cv2.imread(img_path)  # im0.shape like (1028, 1912, 3)      (y,x,c)
    
        patch_transform(patch, (1, 3, im0.shape[1], im0.shape[0]), patch_shape, [im0.shape[1],im0.shape[0]], NUM_PATCH, im0, filename, reg_book, output_dir)

    with open(reg_book_path, 'w') as f:
        json.dump(reg_book, f, indent=4)

if __name__ == "__main__":
    image_input_dir = "/home/aikedaer/mydata/data/comp/images/phase2"
    patch_output_dir = "/home/aikedaer/mydata/data/comp/images/phase2_patch/background_patch"
    reg_book_output_path = "/home/aikedaer/mydata/data/comp/images/phase2_patch/bkg_reg_book.json"
    os.makedirs(patch_output_dir, exist_ok=True)
    extra_patch(image_input_dir, patch_output_dir, reg_book_output_path)