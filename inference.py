#!/usr/bin/env python3
# -*-coding: utf8-*-

import torch
import cv2

SOURCE = 0  # camera index or 'path_to_file'

# Model
model = torch.hub.load('yolov5', 'custom', path='ppe.pt', source='local')

# Model-configuration
model.conf = 0.5  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

print('\n[DETECTOR] START SERVICE...\n')

cap = cv2.VideoCapture(SOURCE)

try:
    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        results = model(img)

        results.render()  # or .print(), .show(), .save(), .crop(), .pandas()

        cv2.imshow('VIDEO', img)

        pressed = cv2.waitKey(1)
        if pressed == 27:
            break
        elif pressed == 32:
            cv2.waitKey(0)

except KeyboardInterrupt:
    print('CTRL+C is pressed.\n')

print('[DETECTOR] FINISH SERVICE.\n')
cv2.destroyAllWindows()
cap.release()
