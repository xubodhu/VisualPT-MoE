import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import torch
from PIL import Image
from torchvision import transforms
import argparse
from pytorch_models.detection import fasterrcnn_resnet50_fpn

'''
存储图像中物体标签
'''

COCO_INSTANCE_CATEGORY_NAMES = [
    'None', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'None', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'None', 'backpack', 'umbrella', 'None', 'None',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'None', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'None', 'dining table',
    'None', 'None', 'toilet', 'None', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'None', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

index_category_dict = {i:cat for i, cat in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', default='./data/MNRE_images/')
    parser.add_argument('--tags_file', default='./ner_img/object_tags_mre.pt',
                        type=str, help='The location of the image features after resnet processing')
    parser.add_argument('--pretrained_model', default='../pretrained_model/fasterrcnn_resnet50_fpn_coco.pth')

    args = parser.parse_args()

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(args.pretrained_model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    img_id_list = os.listdir(args.img_dir)
    count = 0
    img_object_tags_dict = {}

    

    for i, img_id in enumerate(img_id_list):
        image = None
        try:
            img_path = os.path.join(args.img_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
        except:
            img_path = os.path.join(args.img_dir, 'inf.png')
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            count += 1

        image = image.to(device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            predictions = model(image.to(device))[0]
            labels = predictions['labels']
            scores = predictions['scores']
            score_index = scores >= 0.5
            labels_selected = labels[score_index].cpu().numpy()
            object_tags = []
            for label_index in labels_selected:
                tag = index_category_dict[label_index]
                object_tags.append(tag)

            img_object_tags_dict[img_id] = object_tags

    print("count:", count)
    torch.save(img_object_tags_dict, args.tags_file)









