from torchvision.models import resnet50


def resnet_v1_5_to_v1(net):
    net.layer2[0].conv1.stride = (2, 2)
    net.layer2[0].conv2.stride = (1, 1)

    net.layer3[0].conv1.stride = (2, 2)
    net.layer3[0].conv2.stride = (1, 1)

    net.layer4[0].conv1.stride = (2, 2)
    net.layer4[0].conv2.stride = (1, 1)
