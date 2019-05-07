'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    # 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGGStudent(VGG):

    REDUCE_FACTOR = 2

    def __init__(self, vgg_name):
        super(VGGStudent, self).__init__(vgg_name)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                # add back a conpensation convolution network to make block output
                # consistent
                layers += [
                    nn.Conv2d(in_channels,
                        in_channels*self.REDUCE_FACTOR, kernel_size=1, padding=0),
                    nn.BatchNorm2d(in_channels*self.REDUCE_FACTOR),
                    nn.ReLU(inplace=True)
                ]

                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # adjust the next layer input channels
                in_channels = in_channels*self.REDUCE_FACTOR
            else:
                out_channels = x // self.REDUCE_FACTOR
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
        return nn.Sequential(*layers)

# ============================================================================
class AuxiliaryVgg(nn.Module):
    """
    The auxiliary function
    """
    REDUCE_FACTOR = 2

    def __init__(self, vgg_name, phase_idx):
        super().__init__()
        self.vgg_name = vgg_name
        self.phase_idx = phase_idx

        # build the network
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

        # build the features
        self._build_features(cfg[vgg_name])

        # freeze the layers
        self._freeze_all_layers()

        # defreeze the block we want to train
        self._defreeze_target_block()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) # reshape the output
        out = self.classifier(out)
        return out

    def _defreeze_target_block(self):
        if self.phase_idx == 0:
            # this is the teacher network, nothing to defreeze
            return

        block_bnd_idx = [0]
        for l_idx, f in enumerate(self.features):
            if isinstance(f, nn.MaxPool2d):
                block_bnd_idx.append(l_idx)
                # TODO: early complete the linear search

        # =====================================
        # consider the block start index and end index
        blk_start = block_bnd_idx[self.phase_idx-1]
        blk_end = block_bnd_idx[self.phase_idx]

        # defreeze the feature layers
        for f in self.features[blk_start:blk_end]:
            for p in f.parameters():
                p.requires_grad = True

    def _freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_weight_from_last_phase(self, aux_vgg_net):
        pass

    def _build_features(self, cfg):
        # get the block-wise configuration
        student_cfg, teacher_cfg = self._splite_cfg(cfg, self.phase_idx)

        student_layers, channels = self._make_student_layers(student_cfg, 3)

        teacher_layers = self._make_teacher_layers(teacher_cfg, channels)

        # set-up the features
        self.features = nn.Sequential(*student_layers, *teacher_layers)

    @staticmethod
    def _make_teacher_layers(cfg, in_channels):
        layers = []
        channels = in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                channels = x
        return layers

    def _make_student_layers(self, cfg, in_channels):
        layers = []
        channels = in_channels
        for x in cfg:
            if x == 'M':
                # add back a conpensation convolution network to make block output
                # consistent
                layers += [
                    nn.Conv2d(channels, channels*self.REDUCE_FACTOR, kernel_size=1,
                              padding=0),
                    nn.BatchNorm2d(channels*self.REDUCE_FACTOR),
                    nn.ReLU(inplace=True)
                ]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # adjust the next layer input channels
                channels = channels*self.REDUCE_FACTOR
            else:
                out_channels = x // self.REDUCE_FACTOR
                layers += [nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
                channels = out_channels
        return layers, channels

    @staticmethod
    def _splite_cfg(cfg, k=0):
        """
        Split the configuration into teacher subblocks and student subblocks

        Return:
            a tuple of the configuration of teacher subnet and student subnet
        """

        split_idx = 0
        max_pool_cnt = 0
        for idx, l in enumerate(cfg):
            if l == 'M':
                max_pool_cnt += 1
                if max_pool_cnt == k:
                    split_idx = idx+1

        # splite the configuration
        student_cfg = cfg[:split_idx]
        teacher_cfg = cfg[split_idx:]
        return student_cfg, teacher_cfg


if __name__ == "__main__":
    teacher = VGG('VGG16')
    student = VGGStudent('VGG16')
    x = torch.randn(2,3,32,32)
    y1 = teacher(x)
    y2 = student(x)
    assert y1.shape == y2.shape

    def get_sum_params(model):
        ret = 0
        for p in model.parameters():
            ret += p.numel()
        return ret
    print("# of params in teacher submodel:", get_sum_params(teacher))
    print("# of params in student submodel:", get_sum_params(student))
