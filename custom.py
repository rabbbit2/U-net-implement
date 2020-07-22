#%%
import os
import os.path
import torchvision
from PIL import Image
import torchvision.transforms.functional as F

#%%

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(directory, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    for data_name in os.scandir(os.path.join(directory, "images")):
        item= os.path.join(directory,"images",data_name.name),os.path.join(directory,"segmentations",data_name.name)
        instances.append(item)

    return instances

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class segFolder(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, both_transform=None, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform,target_transform=target_transform)
        samples = make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.both_transform = both_transform
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_path, seg_path = self.samples[index]
        sample = self.loader(image_path)
        target = self.loader(seg_path)
        if self.both_transform is not None:
            sample, target=self.both_transform([sample, target])
        if self.transform is not None:
            sample = self.transform(sample)#.cuda()
        if self.target_transform is not None:
            target = self.target_transform(target)#.cuda()
            '''
            ratio = numpy.sum(label)
            weight1 = (numpy.exp(-(ndimage.distance_transform_edt(label)**2)/50)*10+(250000)/(2*ratio))*label

            label = 1-label
            ratio = numpy.sum(label)
            weight2 = (numpy.exp(-(ndimage.distance_transform_edt(label)**2)/50)*10+(250000)/(2*ratio))*label

            weight = torchvision.transforms.ToTensor()(numpy.stack((weight1,weight2),2)).cuda()
            '''
        return sample, target

class Compose(object):

    '''
    連續對於k張圖片同時變換
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t,list):
                for i, k in enumerate(t):
                    if k!=None:
                        img[i] = k(img[i])
            else :
                img=t(img)
        return img

class RandomCrop(torchvision.transforms.RandomCrop):

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image):List of images to be cropped.

        Returns:
            PIL Image: Cropped images.
        """
        if self.padding is not None:
            for i, img in enumerate(imgs):
                imgs[i] = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            for i, img in enumerate(imgs):
                imgs[i] = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            for i, img in enumerate(imgs):
                imgs[i] = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(imgs[0], self.size)

        for index, img in enumerate(imgs):
            imgs[index] = F.crop(img, i, j, h, w)

        return imgs
