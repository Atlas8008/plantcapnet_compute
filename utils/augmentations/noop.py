from torchvision import transforms

#noop_transform = lambda: transforms.Lambda(lambda v: v)

def identity(v):
    return v

def noop_transform():
    return transforms.Lambda(identity)
