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



class Vgg(nn.Module):
    def __init__(self, vgg_name):
        super().__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)

        #
        self._cross_entropy_loss_fn = nn.CrossEntropyLoss()

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
        return nn.Sequential(*layers)

    def get_loss(self, outputs, labels):
        ret = self._cross_entropy_loss_fn(outputs, labels)
        return ret

class VggStudent(Vgg):

    REDUCE_FACTOR = 2

    def __init__(self, vgg_name):
        super().__init__(vgg_name)

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

    def __init__(self, teacher_model, phase_idx, alpha=0.1):
        assert phase_idx > 0
        super().__init__()
        assert isinstance(teacher_model, nn.Module)

        #
        self._cross_entropy_loss_fn = nn.CrossEntropyLoss()

        self.alpha = alpha

        self.vgg_name = teacher_model.vgg_name
        self.phase_idx = phase_idx

        # ================================================================
        # build the network
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, 10)
        # build the features
        self._build_features(cfg[self.vgg_name])
        # ===========================================================
        # ===========================================================
        # obtain the block indices
        self._create_blk_idxs()

        #
        self._set_intercept_layer_idx()

        # set teacher subnetwork block
        self._set_teacher_blk_idxs(teacher_model)
        self._set_teacher_subnet_blk(teacher_model)
        self._transfer_teacher_weights(teacher_model)

        # freeze the layers
        self._freeze_all_layers()
        # defreeze the block we want to train
        self._defreeze_target_block()

    def _transfer_teacher_weights(self, teacher):
        num_blks = len(self._block_bnd_idx)
        # blockwise copying
        for blk in range(1, num_blks):
            if blk == self.phase_idx:
                # no copying when we checking the phase index
                continue
            # get student and teacher staring and ending index of a block
            ss, es = self._block_bnd_idx[blk-1], self._block_bnd_idx[blk]
            st, et = self._teacher_blk_idxs[blk-1], self._teacher_blk_idxs[blk]
            #
            for i, j in zip(range(ss, es), range(st, et)):
                # TODO: think of transfering values, check if all in cpu
                self.features[i] = teacher.features[j]

        # copy classification layers
        self.classifier = teacher.classifier

    def _set_teacher_blk_idxs(self, teacher):
        self._teacher_blk_idxs = [0]
        for l_idx, f in enumerate(teacher.features):
            if isinstance(f, nn.MaxPool2d):
                self._teacher_blk_idxs.append(l_idx)
        # just for checking
        assert len(self._block_bnd_idx) == len(self._teacher_blk_idxs)

    def _set_teacher_subnet_blk(self, teacher):
        blk_end_idx = self._teacher_blk_idxs[self.phase_idx]
        self._teacher_sub_blk = teacher.features[:blk_end_idx]
        for p in self._teacher_sub_blk.parameters():
            p.requires_grad = False

    def drop_teacher_subnet_blk(self):
        self._teacher_sub_blk = None

    def forward(self, x):
        # calcualte the teacher sub-block output
        self.teacher_blk_output = self._teacher_sub_blk(x)

        # apply forwarding from 0 to _intercept_layer_idx and store it
        self.student_blk_output = self.features[:self._intercept_layer_idx](x)
        out = self.features[self._intercept_layer_idx:](self.student_blk_output)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) # reshape the output
        out = self.classifier(out)

        return out

    def get_loss(self, outputs, labels):
        """
        Compute the cross entropy loss given outputs and labels.
        Args:
            outputs: (Variable) dimension batch_size x 6 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
        Returns:
            loss (Variable): cross entropy loss for all images in the batch
        Note:
            you may use a standard loss function from
            http://pytorch.org/docs/master/nn.html#loss-functions.
            This example
            demonstrates how you can easily define a custom loss function.
        """
        # calculate the local loss
        diff = self.teacher_blk_output - self.student_blk_output
        diff = diff.view(diff.size(0), -1)  # flatten
        local_loss = torch.norm(diff, p='fro', dim=1)
        batch_local_loss = torch.mean(local_loss)
        ret = self._cross_entropy_loss_fn(outputs, labels) + self.alpha*batch_local_loss
        return ret

    def _set_intercept_layer_idx(self):
        self._intercept_layer_idx = self._block_bnd_idx[self.phase_idx] - 1

    def _create_blk_idxs(self):
        self._block_bnd_idx = [0]
        for l_idx, f in enumerate(self.features):
            if isinstance(f, nn.MaxPool2d):
                self._block_bnd_idx.append(l_idx)

    def _defreeze_target_block(self):
        if self.phase_idx == 0:
            # this is the teacher network, nothing to defreeze
            return
        # =====================================
        # consider the block start index and end index
        blk_start = self._block_bnd_idx[self.phase_idx-1]
        blk_end = self._block_bnd_idx[self.phase_idx]

        # defreeze the feature layers
        for f in self.features[blk_start:blk_end]:
            for p in f.parameters():
                p.requires_grad = True

    def _freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

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
    teacher = Vgg('VGG16')
    student = VggStudent('VGG16')
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
