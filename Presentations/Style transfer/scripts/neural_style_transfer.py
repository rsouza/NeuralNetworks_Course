import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from torchvision.models import vgg19
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as pyplot


class vgg19_mod(nn.Module):
    def __init__(self, content_layers, style_layers):
        '''
        Class that initialize the VGG19 model and set the intermediate style and content layers

        Inputs:
            style_layers : list of index of the style layers in the VGG19 net
            content_layers: list of index of the content layers in the VGG19 net
        '''
        super(vgg19_mod, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = nn.ModuleList(features).eval()
        self.style_layers = style_layers
        self.content_layers = content_layers

    def forward(self,  input):
        '''
        Forward process that pass the input image trought the VGG 19 net and keeps the intermediates outputs

        Inputs:
            input : tensor of shape [batch, 3, 224, 224] 

        Outputs:
            style_out : list of style layers outputs
            cotent_out : list of content layers outputs
        '''
        style_out = []
        content_out = []
        for i, model in enumerate(self.features):
            input = model(input)
            if i in self.style_layers:
                style_out.append(input)
            if i in self.content_layers:
                content_out.append(input)
        return content_out, style_out

class style_transfer():
    ''' 
    Class that encapsulates the style transfer procedure

    Inputs:
        content : content image, can be a tensor or a path for a image
        style : style image, can be a tensor or a path for a image
        iterations: number of iterations to converge
        content_layers : list of index of VGG19 content layers
        style_layers : list of index of VGG19 style layers
        content_weight : weight of the content loss
        style_weight : weight of the style loss
        verbose : if the function will print intermediate steps


    '''
    def __init__(self, 
                content, 
                style,
                output_path = 'output.jpg',
                content_layers = None, 
                style_layers = None,
                iterations = 500, 
                content_weight = 1, 
                style_weight = 1000000, 
                verbose = False, 
                cuda = False):
        self.content = content
        self.style = style
        self.output_path = output_path
        self.preprocess = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225]),])
        self.verbose = verbose
        self.iterations = iterations
        #Setting intermediate layers
        self.content_layers = content_layers
        if self.content_layers is None:
            self.content_layers = [9]
        self.style_layers = style_layers
        if self.style_layers is None:
            self.style_layers = [0, 3, 6, 9, 14]   

        self.content_weight = content_weight
        self.style_weight = style_weight
        #Setting culda
        if cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')      


    def load_image(self, path):
        '''
        Function that load the image and preprocess to the VGG19 format
        The image must be with shape 224x224, with mean [0.485, 0.456, 0.406]
        and standard deviation [0.229, 0.224, 0.225]
        '''
        if isinstance(path, str):
            img = Image.open(path)
        else:
            img = path
        return self.preprocess(img).unsqueeze(0)

    def content_loss(self, pred, target):
        '''The content loss is the mean squared error'''
        return torch.pow(pred - target, 2).mean()

    def gram_matrice(self, input):
        '''
        Compute the gram matrice of the input tensor
        The gram matrice of V is V'V, but we have to change the input to 2 dimensios
        Than we normalize by the number of elements
        '''
        batch, channel, width, height = input.size()
        M = input.view(batch * channel, width * height)
        gram = torch.mm(M, M.t())
        return gram.div(batch*width*height*channel)

    def style_loss(self, pred, gram_target):
        '''The style loss if the euclidian distance of the Gram matrice '''
        gram_pred = self.gram_matrice(pred)
        return torch.pow(gram_pred - gram_target, 2).mean()

    def train(self):
        '''
        Training process for the neural style transfer VGG19
        The method uses two images, the content and the style image, and generate a image that 
        contain the content and the style of the reespective images
        The optimization is done by LBFGS

        Outputs:
            init_image : result image with content and style

        ''' 

        #Loading the model
        model =  vgg19_mod(self.content_layers, self.style_layers).to(self.device)
            
        if self.verbose:
            print('Initialized model.')

        #Loading the images as tensors
        content_img = self.load_image(self.content).to(self.device)
        style_img = self.load_image(self.style).to(self.device)
        init_img = self.load_image(self.content).to(self.device)
        if self.verbose:
            print('Initialized images.')

        #Getting the content and style outputs
        content_out, _ = model(content_img)
        _, style_out = model(style_img)
        if self.verbose:
            print('Calculated images outputs.')

        #Pre-calculating the gram matrice for the styles outputs
        gram_out = [self.gram_matrice(out) for out in style_out]

        #Setting optmizer
        optmizer = optim.LBFGS([init_img.requires_grad_()])
        if self.verbose:
            print('Starting optimization.')
        for iter in tqdm(range(self.iterations)):

            def closure():
                '''Calculate ouputs, loss and gradients'''
                optmizer.zero_grad()

                init_content_out, init_style_out = model(init_img)
                
                _content_loss = 0
                for l in range(len(content_out)):
                    _content_loss += self.content_loss(init_content_out[l], content_out[l])

                _style_loss = 0
                for l in range(len(style_out)):
                    _style_loss += self.style_loss(init_style_out[l],  gram_out[l])

                _content_loss *= self.content_weight
                _style_loss *= self.style_weight

                loss = _content_loss + _style_loss
                loss.backward(retain_graph=True)
                #print(init_img.grad)

                if iter % 100 == 0 and iter > 0:
                    if self.verbose:
                        print('Iteration %d.  Model loss %.8f'%(iter, loss))
                        print('Content loss: %.8f | Style loss: %.8f'%(_content_loss, _style_loss))
                        print()

                return _content_loss + _style_loss
            optmizer.step(closure)

        #Invert VGG19 input transformation
        invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ])])
        inv_img = invTrans(init_img.squeeze(0))
        save_image(inv_img, self.output_path)
        return inv_img