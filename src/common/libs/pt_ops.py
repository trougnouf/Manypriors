import torch

def pt_crop_batch(batch, cs: int):
    '''
    center crop an image batch to cs
    also compatible with numpy tensors
    '''
    x0 = (batch.shape[3]-cs)//2
    y0 = (batch.shape[2]-cs)//2
    return batch[:, :, y0:y0+cs, x0:x0+cs]

def crop_to_multiple(tensor, multiple=64):
    return tensor[...,:tensor.size(-2)-tensor.size(-2)%multiple,:tensor.size(-1)-tensor.size(-1)%multiple]

def img_to_batch(img, patch_size: int, nchans_per_prior: int = None):
    _, ch, height, width = img.shape
    if nchans_per_prior is None:
        nchans_per_prior = ch
    assert height%patch_size == 0 and width%patch_size == 0, (
        'img_to_batch: dims must be dividable by patch_size. {}%{}!=0'.format(
            img.shape, patch_size))
    assert img.dim() == 4
    return img.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size).transpose(1,0).reshape(
            ch, -1, patch_size, patch_size).transpose(1,0).reshape(-1, nchans_per_prior, patch_size, patch_size)
            
class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g