import torch
import torch.nn as nn
import numpy
from PIL import Image
from skimage import io
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt

image_dir = 'cat.jpg'

vgg19 = models.vgg19(pretrained="imagenet")

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfgmasks = {
 'many': [ 0,  1,  0,    0,   1,   0,   0,   1,   0,   1,  0,   0,   1,   0,   1,   0,   0,   1,   0,   1,   0],
 'tail': [ 0,  0,  0,    0,   0,   0,   0,   0,   0,   1,  0,   0,   1,   0,   1,   0,   0,   1,   0,   1,   0],
'front': [ 0,  1,  0,    0,   1,   0,   0,   1,   0,   1,  0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
}

class Vgg_features(nn.Module):
    def __init__(self, vgg, name):
        super(Vgg_features, self).__init__()
        self.cfg = cfg['E']
        cfgmask = cfgmasks[name]
        self.mask = []
        for i in range(len(cfgmask)):
            if self.cfg[i]=='M':
                #max pooling
                self.mask.append(cfgmask[i])
            else:
                #convolution: consider the map after ReLu
                self.mask+=[0, cfgmask[i]]
        self.features = list(vgg.features.children())
        for f in self.features:
            f.requires_grad = False
        self.length = len(self.features)
        assert self.length == len(self.mask), \
                "network sizes not match {} != {}".format(self.length, len(self.mask))
    
    def forward(self, x):
        output = []
        for i in range(self.length):
            x = self.features[i](x)
            if self.mask[i]==1:
                output.append(x)
        return output

vgg_many = Vgg_features(vgg19, 'many')
vgg_tail = Vgg_features(vgg19, 'tail')
vgg_front =Vgg_features(vgg19, 'front')

def channels_to_map(channels):
    m = channels.mean(0)
    return m

def show_tensor(t):
    a = t.tolist()
    plt.imshow(a)
    plt.show()

data_transform = transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def mean_normalize(v, mean):
    m = v.mean()
    return v*mean/m

def range_normalize(v, a, b):
    mi = v.min()
    ma = v.max()
    return (v - mi)/(ma-mi)*(b-a) + a


def project(m, shape):
    # project one channel map m to a new map of shape=(H1,W1)
    pil = Image.fromarray(m)
    img = transforms.Resize(shape)(pil)
    return numpy.array(img)

def get_energy_map(vgg_features):
    def inner(img_rgb, show=False):
        #input: rgb image, format: h*w*3
        #output: h*w numpy array of numpy.float64
        img = data_transform(img_rgb)
        img = torch.tensor(img, dtype=torch.float)
        c, h, w = img.shape
        assert h>=80 and w>=80, "image too smal -- feature_map.energy_map: {}*{}".format(h, w)
        imgs = torch.stack([img])
        all_features = vgg_features(imgs)
        np_features = [ all[0].detach().numpy() for all in all_features ]
        L = len(np_features)
        maps = [channels_to_map(f) for f in np_features]
        #print("got features")
        """
        sizes= [m.shape for m in maps]
        vecs = [m.view(-1) for m in maps]
        normalized_vecs = [mean_normalize(vecs[i], 1.0) for i in range(L)]
        normalized_maps = [normalized_vecs[i].reshape(sizes[i]) for i in range(L)]
        """
        normalized_maps = [range_normalize(m, 0, 1) for m in maps]
        projected_maps = [project(m, (h, w)) for m in normalized_maps]
        #print("projected")
        sum_map = sum(projected_maps)
        mean_map = mean_normalize(sum_map, 1.0)
        res = numpy.array(mean_map, dtype=numpy.float64)

        if not show:
            return res
    
        plt.figure(1)
        for i in range(L):
            pm = projected_maps[i]
            nm = normalized_maps[i]
            plt.subplot(1, L, i+1)
            print(nm.shape)
            print("  mean pm nm:",pm.mean(), nm.mean())
            print("  pm min:", pm.min(), "   nm min:", nm.min())
            print("  pm max:", pm.max(), "   nm max:", nm.max())
            print("  sum pm nm:", pm.sum(), nm.sum())
            plt.imshow(nm.tolist())
        plt.figure(2)
        plt.imshow(res.tolist())
        plt.show()
        return res
    return inner

energy_map = get_energy_map(vgg_many)
tail_map = get_energy_map(vgg_tail)
front_map = get_energy_map(vgg_front)

## io image format: h*w*3
def main():
    img = io.imread(image_dir)
    print(img.shape)
    energy_map(img, show=True)

if __name__=="__main__":
    main()