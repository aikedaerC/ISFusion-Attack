
    def box_label_center(self, box, img_color, real_label="", label="", color=(128, 128, 128), txt_color=(255, 255, 255), object_width=None, object_height=None, rotated=False, pos=0):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                # Calculate the center of the rotated box
                center_x = sum([b[0] for b in box]) / 4
                center_y = sum([b[1] for b in box]) / 4
                p1 = (center_x, center_y)
                # NOTE: PIL-version polygon needs tuple type.
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                # Calculate the center of the non-rotated box
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                p1 = (center_x, center_y)
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                # Calculate the position to place the label text
                label_x = center_x - w / 2
                label_y = center_y - h / 2
                outside = label_y - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # Place the label text at the calculated position
                self.draw.text((label_x, label_y), label, fill=txt_color, font=self.font)
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                # NOTE: cv2-version polylines needs np.asarray type.
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                # cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                # Calculate the position to place the label text at the center of the box
                label_x = (p1[0] + p2[0] - w) / 2
                label_y = (p1[1] + p2[1] + h) / 2
                # Increase font size
                # if real_label == "traffic lights":
                if max(object_height, object_width)>224 or min(object_height, object_width)>32:
                    font_scale = 1
                    thick = self.lw//2
                else:
                    font_scale = math.sqrt(object_width**2 + object_height**2) / 150.0
                    thick = self.lw//2
                    print(f"self.font_scale {font_scale}")
                # Place the label text at the calculated position with increased font size
                if pos==9:
                    # 创建文本掩码
                    mask = np.zeros_like(self.im, dtype=np.uint8)
                    # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    # text_x = (self.im.shape[1] - text_size[0]) // 2
                    # text_y = (self.im.shape[0] + text_size[1]) // 2
                    # cv2.putText(mask, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    cv2.putText(
                        mask,
                        label,
                        (int(label_x), int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness=thick, # 1
                        lineType=cv2.LINE_AA,
                    )
                    # 应用掩码创建文本颜色图像
                    # color_img_resized = cv2.resize(img_color, (self.im.shape[1], self.im.shape[0]))
                    # 创建一个渐变图像
                    gradient = np.linspace(0, 255, self.im.shape[1], dtype=np.uint8)
                    gradient = np.tile(gradient, (self.im.shape[0], 1))
                    gradient_img = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
                    # gradient_img = cv2.applyColorMap(gradient, cv2.COLORMAP_HSV) # more color
                    # gradient_img = cv2.flip(gradient_img, 1)
                    # 应用文本掩码
                    text_gradient = cv2.bitwise_and(gradient_img, gradient_img, mask=mask[:, :, 0])
                    # 将带颜色的文本覆盖到原图上
                    # final_img = cv2.addWeighted(self.im, 1, text_gradient, 1, 0)
                    self.im[mask > 0] = text_gradient[mask > 0]
                    # self.im = final_img
                    return self.im
                    # return final_img
                    # text_color = cv2.bitwise_and(color_img_resized, color_img_resized, mask=mask[:,:,0])
                    # import pdb;pdb.set_trace()
                    # final_img = cv2.bitwise_or(self.im, text_color)
                    # return final_img
                
                elif pos==0:
                    cv2.putText(
                        self.im,
                        label,
                        (int(label_x), int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        txt_color,
                        thickness=1, # 1
                        lineType=cv2.LINE_AA,
                    )
                elif pos==1:
                    cv2.putText(
                        self.im,
                        label,
                        (int(label_x), int(label_y)+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        txt_color,
                        thickness=self.lw,
                        lineType=cv2.LINE_AA,
                    )
                elif pos==-1:
                    cv2.putText(
                        self.im,
                        label,
                        (int(label_x), int(label_y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        txt_color,
                        thickness=self.lw,
                        lineType=cv2.LINE_AA,
                    )
                elif pos==2:
                    font_scale = max(object_width, object_height) / 290.0
                    cv2.putText(
                        self.im,
                        label,
                        (int(label_x), int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        txt_color,
                        thickness=self.lw,
                        lineType=cv2.LINE_AA,
                    )
