from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from torchvision.models import resnet50
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.xpu import device
import matplotlib.pyplot as plt

from data.datamgr import SimpleDataManager, SetDataManager

from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from methods.stl_deepbdc import STLDeepBDC
from methods.template import BaselineTrain

from utils import *
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=84, type=int, choices=[84, 224])
parser.add_argument('--dataset', default='mini_imagenet', choices=['mini_imagenet', 'tiered_imagenet', 'cub', 'Cars'])
parser.add_argument('--data_path', default='/home/okc/data/miniImageNet-cam', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18'])
parser.add_argument('--method', default='stl_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet'])

parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=5, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born3-144-SCAttn/last_model.tar', help='meta-trained or pre-trained model .tar file path')
parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')

parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
parser.add_argument('--reduce_dim', default=144, type=int, help='the output dimensions of BDC dimensionality reduction layer')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
parser.add_argument('--img_path', default='/home/okc/data/miniImageNet/train/n01532829/n0153282900000007.jpg')
parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
params = parser.parse_args()
num_gpu = set_gpu(params)

json_file_read = False

base_datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
base_loader = base_datamgr.get_data_loader('train', aug=False)


model = BaselineTrain(params, model_dict[params.model], params.num_classes)
# model save path
model = model.cuda()
model.eval()

modelfile = os.path.join(params.model_path)
tmp = torch.load(modelfile)
state = tmp['state']
model.load_state_dict(state)


# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]

print(model)



transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
            ])
#-----------------------------------------------------
img = Image.open(params.img_path).convert('RGB')
img_np = img.resize((84,84))
img_np = np.array(img_np) / 255.0
img_np = img_np.astype(np.float32)

#targets = [ClassifierOutputTarget(1)]

mean = [0.472, 0.453, 0.410]
std = [0.277, 0.268, 0.285]


def inverse_normalize(tensor, mean, std):
    # 将 mean 和 std 转换为与输入 tensor 相同形状的 tensor
    mean = torch.tensor(mean).view(1, 3, 1, 1)  # 形状调整为 (1, 3, 1, 1)，适用于批处理
    std = torch.tensor(std).view(1, 3, 1, 1)  # 形状调整为 (1, 3, 1, 1)

    # 进行逆归一化
    tensor = tensor * std + mean
    return tensor

for _, (x, y) in enumerate(base_loader):
    target_layers = [model.feature.layer4[-1]]
    targets = [ClassifierOutputTarget(l.item()) for l in y]
    input_tensor = x.cuda()
    inverse_tensor = inverse_normalize(x, mean, std)
    # xnp = inverse_tensor[1].permute(1, 2, 0).numpy()
    # xnp = np.clip(xnp, 0, 1)
    # # xnp = Image.open(base_loader.dataset.data[0]).convert('RGB')
    # # xnp = xnp.resize((84, 84))
    # # xnp = np.array(xnp) / 255.0
    # # xnp = xnp.astype(np.float32)
    # plt.imshow(xnp)
    # plt.axis('off')
    # plt.show()
    # with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     for i in range(len(targets)):
    #         gradcam = grayscale_cam[i, :]
    #         base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
    #         base_img = np.clip(base_img, 0, 1)
    #         visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
    #         visual_pil = Image.fromarray(visual)
    #         base_pil = Image.fromarray((base_img * 255).astype(np.uint8))
    #         visual_pil.save('/home/okc/data/save/%d/%d_res_cam.png' % (y[i].item(), i))
    #         base_pil.save('/home/okc/data/save/%d/%d_ori.png' % (y[i].item(), i))

        # grayscale_cam = grayscale_cam[1, :]
        # visualization = show_cam_on_image(xnp, grayscale_cam, use_rgb=True)
    # plt.imshow(visualization)
    # plt.axis('off')
    # plt.show()

    target_layers = [model.dcov.SCAtten]
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        for i in range(len(targets)):
            gradcam = grayscale_cam[i, :]
            base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
            base_img = np.clip(base_img, 0, 1)
            visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
            visual_pil = Image.fromarray(visual)
            visual_pil.save('/home/okc/data/save/%d/%d_res_sca_cam.png' % (y[i].item(), i))
    #     grayscale_cam = grayscale_cam[1, :]
    #     visualization = show_cam_on_image(xnp, grayscale_cam, use_rgb=True)
    # plt.imshow(visualization)
    # plt.axis('off')
    # plt.show()

    # target_layers = [model.feature.layer1[-1]]
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     for i in range(len(targets)):
    #         gradcam = grayscale_cam[i, :]
    #         base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
    #         base_img = np.clip(base_img, 0, 1)
    #         visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
    #         visual_pil = Image.fromarray(visual)
    #         base_pil = Image.fromarray((base_img * 255).astype(np.uint8))
    #         visual_pil.save('/home/okc/data/save/%d/%d_res_layer1_cam.png' % (y[i].item(), i))
    #         base_pil.save('/home/okc/data/save/%d/%d_ori.png' % (y[i].item(), i))
    # target_layers = [model.feature.layer2[-1]]
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     for i in range(len(targets)):
    #         gradcam = grayscale_cam[i, :]
    #         base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
    #         base_img = np.clip(base_img, 0, 1)
    #         visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
    #         visual_pil = Image.fromarray(visual)
    #         visual_pil.save('/home/okc/data/save/%d/%d_res_layer2_cam.png' % (y[i].item(), i))
    # target_layers = [model.feature.layer3[-1]]
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     for i in range(len(targets)):
    #         gradcam = grayscale_cam[i, :]
    #         base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
    #         base_img = np.clip(base_img, 0, 1)
    #         visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
    #         visual_pil = Image.fromarray(visual)
    #         visual_pil.save('/home/okc/data/save/%d/%d_res_layer3_cam.png' % (y[i].item(), i))
    # target_layers = [model.feature.layer4[-1]]
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     for i in range(len(targets)):
    #         gradcam = grayscale_cam[i, :]
    #         base_img = inverse_tensor[i].permute(1, 2, 0).numpy()
    #         base_img = np.clip(base_img, 0, 1)
    #         visual = show_cam_on_image(base_img, gradcam, use_rgb=True)
    #         visual_pil = Image.fromarray(visual)
    #         visual_pil.save('/home/okc/data/save/%d/%d_res_layer4_cam.png' % (y[i].item(), i))
#-----------------------------------------------------
# test_dataset = datasets.ImageFolder(root='/home/okc/data/miniImageNet-cam/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)
# for x,y in test_loader:
#     input_tensor = x.cuda()
#     with GradCAM(model=model, target_layers=target_layers) as cam:
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
#         model_outputs = cam.outputs
#     plt.imshow(visualization)
#     plt.axis('off')
#     plt.show()
#-----------------------------------------------------

