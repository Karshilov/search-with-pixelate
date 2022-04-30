#%%
from torchvision import transforms 
from torch.utils.data import Dataset
from torch import nn, randint
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from glob import glob
from faker import Faker

fake = Faker(["zh_CN"])

platforms = ['知乎', '微博', '孙笑川吧', '百度图片', 'bili']

aug_images = [Image.open(name).convert("RGBA") for name in glob('./aug/*.png')]

class DianZiBaoJiang(nn.Module):
    def __init__(self, size=(8, 16)):
        super().__init__()
        self.size = size

    def forward(self, image):
        layer = random.randint(1, 4)
        for _ in range(layer):
            image = image.convert("RGBA")
            font = ImageFont.truetype("MSYH.TTC", random.randint(*self.size))
            text_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
            image_draw = ImageDraw.Draw(text_overlay)
            text = random.choice(platforms) + '@' + fake.name()
            text_size_x, text_size_y = font.getsize(text)
            text_xy = (random.randint(0, image.size[0] - text_size_x), random.randint(0, image.size[1] - text_size_y))
            image_draw.text(text_xy, text, font=font, fill=(255, 255, 255, 190))
            image = Image.alpha_composite(image, text_overlay)
        return image.convert("RGB")
#%%
class BiaoQingBao(nn.Module):
    def __init__(self, size=(30, 60)):
        super().__init__()
        self.size = size
    def forward(self, image):
        image = image.convert("RGBA")
        cover = random.choice(aug_images)
        length = random.randint(*self.size)
        cover = transforms.Resize([length, length])(cover)
        coordinates = (random.randint(0, (image.size[0] - cover.size[0])), random.randint(0, (image.size[1] - cover.size[1])))
        _r, _g, _b, a = cover.split()
        image.paste(cover, coordinates, mask=a)
        return image.convert("RGB")

from io import BytesIO
class JPEGCompression(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image: Image)->Image:
        out = BytesIO()
        image.save(out, format='jpeg', quality=random.randint(50, 100))
        image = Image.open(out)
        return image.convert("RGB")
#%%
# image = Image.open('test.png')
# image.show()
# image = JPEGCompression()(image)
# image.show()
#%%
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Danbooru(Dataset):

    def __init__(self, root_dir, transform, pure_transform):
        self.transform = transform
        self.pure_transform = pure_transform
        self.files = glob(f'{root_dir}/**/*.jpg')

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, idx):
        image: Image = Image.open(self.files[idx]).convert("RGB")
        return [self.transform(image), self.pure_transform(image)]
        
