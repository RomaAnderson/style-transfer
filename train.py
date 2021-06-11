import torch
from torchvision import transforms

from matplotlib import pyplot

def prep_data(img_size):
    prep = transforms.Compose([transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    
    return prep

def post():
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                    std=[1,1,1]),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                            ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    return postpa, postpb

def postp(tensor, postpa, postpb): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

def train(opt_img, vgg, weights, loss_fns, targets, optim, loss_layers, postpa, postpb, low_res = True):
    #run style transfer
    max_iter = (500 if low_res else 200)
    show_iter = 50
    optimizer = optim.LBFGS([opt_img])
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0]+=1
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
            return loss
        
        optimizer.step(closure)
        
    #display result
    out_img = postp(opt_img.data[0].cpu().squeeze(), postpa, postpb)
    pyplot.imshow(out_img)
    pyplot.gcf().set_size_inches(10,10)

    return out_img
