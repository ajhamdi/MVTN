import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from .Model import Model
from ..tools.view_gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP 
# from tools.view_gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP ##############


mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class view_GCN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet18', num_views=20):
        super(view_GCN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


        FIXED_VIEWS_NB = 4
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        self.vertices = torch.tensor(vertices).cuda()

        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2)
        if self.num_views == 20 :
            self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4)
        elif self.num_views ==12:
            self.LocalGCN3 = LocalGCN(k=4, n_views=FIXED_VIEWS_NB)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2, nclasses=self.nclasses)
        if self.num_views == 20:
            self.View_selector2 = View_selector(
                n_views=self.num_views//2, sampled_view=self.num_views//4, nclasses=self.nclasses)
        elif self.num_views == 12:
            self.View_selector2 = View_selector(
                n_views=self.num_views//2, sampled_view=FIXED_VIEWS_NB, nclasses=self.nclasses)


        self.cls = nn.Sequential(
            nn.Linear(512*3,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        views = self.num_views
        # N, V, C, H, W = x.size()
        # x = x.contiguous().view(-1, C, H, W)
        y = self.net_1(x)

        y = y.view((int(x.shape[0] / views), views, -1))
        # vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1).to(torch.float32)  ##################
        vertices = self.vertices

        y = self.LocalGCN1(y,vertices)

        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]


        z, F_score, vertices2 = self.View_selector1(
            y2, vertices, k=4)

        z = self.LocalGCN2(z,vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]

        w, F_score2, vertices3 = self.View_selector2(
            z2, vertices2, k=4)
        w = self.LocalGCN3(w,vertices3)
        pooled_view3 = torch.max(w, 1)[0]

        self.pooled_view = torch.cat((pooled_view1,pooled_view2,pooled_view3),1)
        pooled_view = self.cls(self.pooled_view)
        return pooled_view,F_score,F_score2
