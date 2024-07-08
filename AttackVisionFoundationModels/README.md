# Attack-Vison-Foundation-Models

## Getting Started

---

### Installation

The installation of this project is extremely easy. You only need to:

- Configurate the environment, vicuna weights, following the instruction in https://github.com/Vision-CAIR/MiniGPT-4    


### Runing

- Add noise to the original image.
```
python gptv_attack.py --add_noise
```
- Add text to the original image. 

> Note: The text adding function is modified in source code level, so before adding text, please copy the funciotn `box_label_center()` at box_label_center.py into /root/miniconda3/lib/python3.9/site-packages/ultralytics/utils/plotting.py just after `box_label()` function in `Annotator` class.

```
python gptv_attack.py --add_text
```



> For the convinent of calculating SSIM, we also implement a GUI tool which can both process single image pair and batch image pairs.

```sh
python batchSSIM.py
```

![ssimcal](https://github.com/HongdaChen/picx-images-hosting/raw/master/20240521/ssimcal.1hs1hshqcr.webp)
