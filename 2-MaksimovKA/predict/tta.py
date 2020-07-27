import torch


def torch_flipud(x):
    """
    Flip image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x):
    """
    Flip image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def torch_rot90(x):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x):
    return x.flip(2).flip(3)


def torch_rot270(x):
    return x.transpose(2, 3).flip(3)

def flip_image2mask(model, image):
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.
    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = (torch.sigmoid(model(image)) +
              torch.sigmoid(torch_fliplr(model(torch_fliplr(image)))) +
              torch.sigmoid(torch_flipud(model(torch_flipud(image))))
             )
    one_over_3 = float(1.0 / 3.0)
    return output * one_over_3