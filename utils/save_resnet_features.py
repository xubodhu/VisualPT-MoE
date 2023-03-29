import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import argparse

class myResnet(torch.nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        # (3,224,224) -> (64,112,112)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # (64,112,112) -> (64,56,56)
        x = self.resnet.maxpool(x)

        # (64,56,56) -> (256,56,56)
        x = self.resnet.layer1(x)
        # (256,56,56) -> (512,28,28)
        x = self.resnet.layer2(x)
        # (512,28,28) -> (1024,14,14)
        x = self.resnet.layer3(x)
        # (1024,14,14) -> (2048,7,7)
        x = self.resnet.layer4(x)

        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', default='./data/MNRE_images/')
    parser.add_argument('--feature_file', default='./ner_img/resnet50_features_mre.pt',
                        type=str, help='The location of the image features after resnet processing')
    parser.add_argument('--pretrained_model',default='../pretrained_model/resnet152.pth')

    args = parser.parse_args()

    model = torchvision.models.resnet50(pretrained=True)
    # model.load_state_dict(torch.load(args.pretrained_model))
    resnet = myResnet(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    img_features = {}
    img_id_list = os.listdir(args.img_dir)
    count = 0
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
            img_feature = resnet(image)
        img_feature = img_feature.squeeze(0)
        img_features[img_id] = img_feature.to("cpu")
    print("count:", count)
    torch.save(img_features, args.feature_file)











