import torch
import numpy as np


def tensor2im(input_image, imtype=np.uint8, keep_alpha_channel=False, rescale=True):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if rescale:
            # post-processing: tranpose and scaling
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
        else:
            image_numpy = image_numpy * 255

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    # cvt rgba to rgb
    if image_numpy.shape[-1] == 4 and not keep_alpha_channel:
        a = image_numpy[:, :, -1].astype("float") / 255
        a = np.stack([a] * 3, axis=-1)
        img = image_numpy[:, :, :-1]
        white = np.ones_like(img) * 255
        image_numpy = img * (a) + white * (1 - a)

    return image_numpy.astype(imtype)
