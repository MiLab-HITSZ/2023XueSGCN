import torchvision.models as torch_models

from backend.gcn.models.IJCNN.SALGL.resnet_salgl import ResnetSalgl
from backend.gcn.models.IJCNN.KMeans.resnet_kmeans import ResnetKmeans
from backend.gcn.models.IJCNN.Agg_SALGL.resnet_agg_salgl import ResnetAggSalgl
from backend.gcn.models.IJCNN.GCN.c_gcn import ResnetCGCN
from backend.gcn.models.IJCNN.GCN.p_gcn import ResnetPGCN



def resnet_agg_salgl(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetAggSalgl(model, num_scenes=num_scenes, num_classes=num_classes).to(device)


def resnet_salgl(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetSalgl(model, num_scenes=num_scenes, num_classes=num_classes).to(device)


def resnet_kmeans(pretrained, device, num_scenes=6, num_classes=80):
    model = torch_models.resnet101(pretrained=pretrained, num_classes=1000)
    return ResnetKmeans(model, num_scenes=num_scenes, num_classes=num_classes).to(device)



def resnet_c_gcn(pretrained, dataset, t, adjList=None, device='cpu', num_classes=80, in_channel=300):
    model = torch_models.resnet101(pretrained=pretrained)

    model = ResnetCGCN(model=model, num_classes=num_classes, in_channel=in_channel, t=t, adjList=adjList)
    return model.to(device)


def resnet_p_gcn(pretrained, adjList=None, device='cpu', num_classes=80, in_channel=2048, out_channel=1):
    model = torch_models.resnet101(pretrained=pretrained)
    model = ResnetPGCN(model=model, num_classes=num_classes, in_channel=in_channel, out_channels=out_channel,
                       adjList=adjList)
    return model.to(device)
