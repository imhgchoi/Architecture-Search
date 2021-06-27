import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.args = args

        num_blocks = [2,2,2,2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(Bottleneck,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.to(args.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5



class FocalLoss(nn.Module):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self.priors_cxcy = self.coder.center_anchor
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.num_classes = self.coder.num_classes
        self.bce = nn.BCELoss(reduction='none')
        self.smooth_l1 = SmoothL1Loss()
        # self.smooth_l1 = nn.SmoothL1Loss(reduction=None)

    def forward(self, pred, b_boxes, b_labels):
        """
        Forward propagation.
        :param pred (loc, cls) prediction tuple (N, 67995, 4) / (N, 67995, num_classes) or [120087] anchors
        :param labels: true object labels, a list of N tensors
        """
        pred_loc = pred[0]
        pred_cls = pred[1]

        batch_size = pred_loc.size(0)
        n_priors = self.priors_xy.size(0)

        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)                        # (N, 67995, 4)
        true_classes = -1 * torch.ones((batch_size, n_priors, self.num_classes), dtype=torch.float).to(device)  # (N, 67995, num_classes)
        depth = -1 * torch.ones((batch_size, n_priors), dtype=torch.bool).to(device)                            # (N, 67995)

        for i in range(batch_size):
            boxes = b_boxes[i]  # xy coord
            labels = b_labels[i]

            ###################################################
            #           match strategies  -> make target      #
            ###################################################
            iou = find_jaccard_overlap(self.priors_xy, boxes)  # [67995, num_objects]
            IoU_max, IoU_argmax = iou.max(dim=1)               # [67995]

            negative_indices = IoU_max < 0.4

            # =======================  make true classes ========================
            true_classes[i][negative_indices, :] = 0           # make negative

            depth[i][negative_indices] = 0

            positive_indices = IoU_max >= 0.5                  # iou 가 0.5 보다 큰 아이들 - [67995]
            argmax_labels = labels[IoU_argmax]                 # assigned_labels

            # class one-hot encoding
            # 0 으로 만들고 이후에 1 을 넣어주기
            true_classes[i][positive_indices, :] = 0
            true_classes[i][positive_indices, argmax_labels[positive_indices].long()] = 1.  # objects

            depth[i][positive_indices] = 1

            # ===========================  make true locs ===========================
            true_locs_ = xy_to_cxcy(boxes[IoU_argmax])                               # [67995, 4] 0~1 사이이다. boxes 가
            true_locs_ = self.coder.encode(true_locs_)
            true_locs[i] = true_locs_

        # ------------------------------------------ cls loss ------------------------------------------
        alpha = 0.25
        gamma = 2

        alpha_factor = torch.ones_like(true_classes).to(device) * alpha                    # alpha
        a_t = torch.where((true_classes == 1), alpha_factor, 1. - alpha_factor)            # a_t
        p_t = torch.where(true_classes == 1, pred_cls, 1 - pred_cls)                       # p_t
        bce = self.bce(pred_cls, true_classes)
        cls_loss = a_t * (1 - p_t) ** gamma * bce                                          # focal loss

        cls_mask = (depth >= 0).unsqueeze(-1).expand_as(cls_loss)                          # both fore and back ground
        num_of_pos = (depth > 0).sum().float().clamp(min=1)                                # only foreground (min=1)
        cls_loss = (cls_loss * cls_mask).sum() / num_of_pos                                # batch 의 bce loss
                                                                                           # / batch 의 object 총갯수

        # ------------------------------------------ loc loss ------------------------------------------
        loc_mask = (depth > 0).unsqueeze(-1).expand_as(true_locs)                          # only foreground
        loc_loss = self.smooth_l1(pred_loc, true_locs)  # (), scalar
        loc_loss = (loc_mask * loc_loss).sum() / num_of_pos
        # loc_loss *= 2                                                                      # balance values

        total_loss = (cls_loss + loc_loss)
        return total_loss, (loc_loss, cls_loss)