import torch
import torch.nn.functional as F


def process_background(segmentations, input_image, bg_type):
    if bg_type == "zero":
        bg = torch.zeros_like(segmentations[:, 0:1])
        segmentations = torch.cat([segmentations, bg], dim=1)
    elif bg_type == "thresh":
        bg = 0.01 * torch.ones_like(segmentations[:, 0:1])#1 - torch.logsumexp(segmentations * 100, dim=1, keepdim=True) / 100
        segmentations = torch.cat([segmentations, bg], dim=1)
    elif bg_type in ("class", "class_zero", "class_thresh"):
        # BG from the pre-trained model is usually a position 1, so push it to the back
        bg = segmentations[:, 0:1]
        
        if bg_type == "class_zero":
            bg = torch.zeros_like(bg)
        elif bg_type == "class_thresh":
            bg = torch.where(
                torch.all(segmentations[:, 1:] <= 0.5, dim=1, keepdim=True),
                torch.ones_like(bg),
                torch.zeros_like(bg),
            )
            
        segmentations = torch.cat([segmentations[:, 1:], bg], dim=1)
    elif bg_type == "greenthresh":
        thresh = 0.0
        mode = "bgr"
        
        input_image = F.interpolate(
            input_image, 
            size=segmentations.shape[-2:], 
            mode="bilinear"
        )
                    
        # Currently hardcoded inversion of caffe image normalization
        # Image is unnormalized to actually be able to determine the greenness
        means = torch.tensor([103.939, 116.779, 123.68])[None, :, None, None].to(input_image.get_device())
        input_image = input_image + means
        
        if mode == "rgb":
            red = input_image[:, 0:1]
            green = input_image[:, 1:2]
            blue = input_image[:, 2:3]
        elif mode == "bgr":
            red = input_image[:, 2:3]
            green = input_image[:, 1:2]
            blue = input_image[:, 0:1]
        
        r = red / (red + green + blue)
        g = green / (red + green + blue)
        b = blue / (red + green + blue)
        
        bg = torch.where(
            2 * g - r - b <= thresh,
            torch.ones_like(segmentations[:, 0:1]),
            torch.zeros_like(segmentations[:, 0:1]),
        )
        
        segmentations = torch.where(
            2 * g - r - b <= thresh,
            torch.zeros_like(segmentations),
            segmentations,
        )
        
        segmentations = torch.cat([segmentations, bg], dim=1)
        
    return segmentations, bg