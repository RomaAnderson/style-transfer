import os 
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

from PIL import Image

from train import prep_data, post, train
from models import VGG, GramMatrix, GramMSELoss

# if not os.path.exists('/images/'):
#     os.makedirs('/images/')

# if not os.path.exists('/outputs/'):
#     os.makedirs('/outputs/')

# if not os.path.exists('/models/'):
#     os.makedirs('/models/')

image_dir = os.getcwd() + '/images/'
model_dir = os.getcwd() + '/models/'

def main():
    parser = argparse.ArgumentParser(description='Style transfer')
    parser.add_argument('--image', '-i', type=str, default=None, help='image path e.g. image.jpg')
    parser.add_argument('--style', '-s', type=str, default=None, help='style path e.g. picasso.jpg')

    args = parser.parse_args()

    #
    # Initialise
    #

    # ----------------- get data -----------------------------------------------------------
    prep = prep_data(512)
    postpa, postpb = post()

    # ----------------- get model -----------------------------------------------------------
    vgg = VGG()
    vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()
    
    # ----------------- load images -----------------------------------------------------------
    img_dirs = [image_dir, image_dir]
    img_names = [args.style, args.image]
    imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    # ----------------- define layers -----------------------------------------------------------
    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
    #these are good weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    #
    # Low res
    #

    print("processing low res...")

    out_img = train(opt_img, vgg, weights, loss_fns, targets, optim, loss_layers, postpa, postpb, low_res = True)

    #
    # high res
    #

    print("processing high res...")

    #prep hr images
    prep_hr = prep_data(800)
    imgs_torch = [prep_hr(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    #now initialise with upsampled lowres result
    opt_img = prep_hr(out_img).unsqueeze(0)
    opt_img = Variable(opt_img.type_as(content_image.data), requires_grad=True)

    #compute hr targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    out_img_hr = train(opt_img, vgg, weights, loss_fns, targets, optim, loss_layers, postpa, postpb, low_res = False)

    out_img_hr.save(f'outputs/{str(img_names[1]).split(".")[0]}_{str(img_names[0]).split(".")[0]}_out_hr.jpg')

    print(f'output saved to: outputs/{str(img_names[1]).split(".")[0]}_{str(img_names[0]).split(".")[0]}_out_hr.jpg')
    


if __name__ == '__main__':
    main()