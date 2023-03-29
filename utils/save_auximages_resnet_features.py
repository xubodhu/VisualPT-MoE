import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import argparse

# class myResnet(torch.nn.Module):
#     def __init__(self, resnet):
#         super(myResnet, self).__init__()
#         self.resnet = resnet
#
#     def forward(self, x):
#         # (3,224,224) -> (64,112,112)
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         # (64,112,112) -> (64,56,56)
#         x = self.resnet.maxpool(x)
#
#         # (64,56,56) -> (256,56,56)
#         x = self.resnet.layer1(x)
#         # (256,56,56) -> (512,28,28)
#         x = self.resnet.layer2(x)
#         # (512,28,28) -> (1024,14,14)
#         x = self.resnet.layer3(x)
#         # (1024,14,14) -> (2048,7,7)
#         x = self.resnet.layer4(x)
#
#         return x

class myResnet(torch.nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            # (bsz, 256, 56, 56)
            x = layer(x)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                # (bsz, channels, 2, 2)
                prompt_kv = torch.nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)
                # [(batch_size, 256, 2, 2), (batch_size, 512, 2, 2), (batch_size, 1024, 2, 2), (batch_size, 2048, 2, 2)]
                prompt_guids.append(prompt_kv)
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, 4, -1)
        return prompt_guids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', default='./data/MNRE_images/')
    parser.add_argument('--feature_file', default='./ner_img/resnet50_features_aux_mre.pt',
                        type=str, help='The location of the image features after resnet processing')
    parser.add_argument('--pretrained_model',default='../pretrained_model/resnet50.pth')

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

    aux_img_train_path = '../HVPNet/data/RE_data/img_vg/train/crops/'
    aux_img_dev_path = '../HVPNet/data/RE_data/img_vg/val/crops/'
    aux_img_test_path = '../HVPNet/data/RE_data/img_vg/test/crops/'

    aux_img_train_dict_path = '../HVPNet/data/RE_data/txt/mre_train_dict.pth'
    aux_img_dev_dict_path = '../HVPNet/data/RE_data/txt/mre_dev_dict.pth'
    aux_img_test_dict_path = '../HVPNet/data/RE_data/txt/mre_test_dict.pth'

    train_data = open('../HVPNet/data/RE_data/txt/ours_train.txt', 'r', encoding='utf8').readlines()
    dev_data = open('../HVPNet/data/RE_data/txt/ours_val.txt', 'r', encoding='utf8').readlines()
    test_data = open('../HVPNet/data/RE_data/txt/ours_test.txt', 'r', encoding='utf8').readlines()

    aux_img_train_dict = torch.load(aux_img_train_dict_path)
    aux_img_dev_dict = torch.load(aux_img_dev_dict_path)
    aux_img_test_dict = torch.load(aux_img_test_dict_path)

    img_features = {}

    count = 0

    for i, (k, v) in enumerate(aux_img_train_dict.items()):
        aux_imgs = []
        for v_ in v:
            if len(aux_imgs) >= 3:
                break
            image = None
            try:
                img_path = os.path.join(aux_img_train_path, v_)
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
                aux_imgs.append(img_feature)

        for j in range(3 - len(aux_imgs)):
            aux_imgs.append(torch.zeros((4, 3840)).to(device))

        img_features[i] = torch.cat(aux_imgs, dim=0).to("cpu")

    print("count:", count)
    torch.save(img_features, './ner_img/resnet50_features_aux_mre_train.pt')
    img_features = {}



    for i, (k, v) in enumerate(aux_img_dev_dict.items()):
        aux_imgs = []
        for v_ in v:
            if len(aux_imgs) >= 3:
                break
            image = None
            try:
                img_path = os.path.join(aux_img_dev_path, v_)
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
                aux_imgs.append(img_feature)

        for j in range(3 - len(aux_imgs)):
            aux_imgs.append(torch.zeros((4, 3840)).to(device))

        img_features[i] = torch.cat(aux_imgs, dim=0).to("cpu")

    print("count:", count)
    torch.save(img_features, './ner_img/resnet50_features_aux_mre_dev.pt')
    img_features = {}


    for i, (k, v) in enumerate(aux_img_test_dict.items()):
        aux_imgs = []
        for v_ in v:
            if len(aux_imgs) >= 3:
                break
            image = None
            try:
                img_path = os.path.join(aux_img_test_path, v_)
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
                aux_imgs.append(img_feature)

        for j in range(3 - len(aux_imgs)):
            aux_imgs.append(torch.zeros((4, 3840)).to(device))

        img_features[i] = torch.cat(aux_imgs, dim=0).to("cpu")

    print("count:", count)
    torch.save(img_features, './ner_img/resnet50_features_aux_mre_test.pt')
    img_features = {}












