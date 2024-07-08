
import cv2
from tqdm import tqdm
import os
import json
from torchvision.transforms import functional as TF
import torchvision


def resize_to_original(image_tensor, original_size):
    # Assuming image_tensor is a PyTorch tensor in C x H x W format
    return TF.resize(image_tensor, original_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)


def pin_back(img_dir, reg_book, new_folder, out_dir):
    for filename in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, filename)
        im0 = cv2.imread(img_path)  # im0.shape like (1028, 1912, 3)
        bbox = reg_book[filename]["bbox"]
        # import pdb;pdb.set_trace()
        obj_path = reg_book[filename]["obj"]
        exist_obj = reg_book[filename]["exist"]
        for idx in range(len(obj_path)):
            # if exist_obj[idx] == "NO":
            #     continue
            # obj = cv2.imread(obj_path[idx].replace("detected_obj","attacked_obj")) 
            fname = os.path.basename(obj_path[idx])
            new_path = os.path.join(new_folder, fname)
            obj = cv2.imread(new_path) 
            try: 
                tp = im0[int(bbox[idx][1]):int(bbox[idx][3]), int(bbox[idx][0]):int(bbox[idx][2])]
                
                im0[int(bbox[idx][1]):int(bbox[idx][3]), int(bbox[idx][0]):int(bbox[idx][2])] = obj #resize_to_original(obj,(tp.shape[1], tp.shape[0]))
            except:
                import pdb;pdb.set_trace()
                print(f"error pin")
                continue
        cv2.imwrite(os.path.join(out_dir, filename), im0)


if __name__ == "__main__":
    original_image_dir = "/home/aikedaer/mydata/data/comp/images/phase2"
    input_reg_book_path = "/home/aikedaer/mydata/data/comp/images/phase2_patch/bkg_reg_book.json" # "reg_book.json"

    INPUT_AFTER_ATTACKED_FLODER = "/home/aikedaer/mydata/data/comp/attacked/phase2/unidiff_minigpt4/background"
    output_dir = "/home/aikedaer/mydata/data/comp/attacked/phase2/attackVLM_minigpt4_patch_together/"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_reg_book_path, 'r') as f:
        reg_book = json.load(f)
    pin_back(original_image_dir, reg_book, INPUT_AFTER_ATTACKED_FLODER, out_dir=output_dir)

