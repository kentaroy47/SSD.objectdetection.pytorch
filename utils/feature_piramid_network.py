# TBD

import torch.nn as nn

from ssd import make_vgg, make_extras, L2Norm

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        # call SSD network
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])
        
        # make Dbox
        dbox = DBox(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
            
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        
        # VGGのconv4_3まで計算
        for k in range(23):
            x = self.vgg[k](x)
        
        # conv4_3の出力をL2Normに入力。source1をsourceに追加
        source1 = self.L2Norm(x)
        sources.append(source1)
        
        # VGGを最後まで計算しsource2を取得
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        
        sources.append(x)
        
        # extra層の計算を行う。
        # source3-6に結果を格納。
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace = True)
            if k % 2 == 1:
                sources.append(x)
        
        # source 1-6にそれぞれ対応するconvを適応しconfとlocを得る。
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # convの出力は[batch, 4*anker, fh, fw]なので整形しなければならない。
        # まず[batch, fh, fw, anker]に整形
        
        # locとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # これで後段の処理につっこめるかたちになる。
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            # Detectのforward
            return self.detect(output[0], output[1], output[2].to(self.device))
        else:
            return output
