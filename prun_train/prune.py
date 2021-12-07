from yolox.exp import get_exp
import torch
from copy import deepcopy
import numpy as np
import torch.nn as nn
from tools.train import make_parser
import time

from prune_model import yolox_prun, YOLOXHead_prune,Bottleneck_prune
from my_train import MY_Trainer_Fine
from yolox.models.network_blocks import Bottleneck


def gen_bn_module(model):
    ignore_bn_list = []
    prune_bn_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, Bottleneck_prune):
            if layer.use_add:
                ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv1.bn")
                ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv2.bn")
                ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv3.bn")
                ignore_bn_list.append(name + '.conv1.bn')
                ignore_bn_list.append(name + '.conv2.bn')
        if "head" in name:  # head preds
        # if "cls_preds" in name or "reg_preds" in name or "obj_preds" in name:  # head preds
            ignore_bn_list.append(name)
        if isinstance(layer, nn.BatchNorm2d):
            if name not in ignore_bn_list:
                prune_bn_dict[name] = layer
    return prune_bn_dict, ignore_bn_list


def gather_bn_weights(prune_bn_dict):
    size_list = [idx.weight.data.shape[0] for idx in prune_bn_dict.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(prune_bn_dict.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights


def obtain_filters_mask(prune_bn_dict, thresh):
    mask_dict = {}
    total = 0
    pruned = 0
    for k, v in prune_bn_dict.items():
        weight_copy = v.weight.data.abs().clone()
        channels = weight_copy.shape[0]
        min_channel_num = int(channels * 0.5) if int(channels * 0.5) > 0 else 1
        mask = weight_copy.gt(thresh).float()
        if int(torch.sum(mask)) < min_channel_num:
            _, sorted_index_weights = torch.sort(weight_copy, descending=True)
            mask[sorted_index_weights[:min_channel_num]] = 1.
        remain = int(mask.sum())
        pruned = pruned + mask.shape[0] - remain

        total += mask.shape[0]
        mask_dict[k] = mask.clone()

    prune_ratio = pruned / total
    return prune_ratio, mask_dict


def prune_model_keep_size(model, mask_dict):

    losses_model = deepcopy(model)
    prune_bn_dict, _ = gen_bn_module(losses_model)
    for mask, bn_module in zip(mask_dict.values(), prune_bn_dict.values()):
        bn_module.weight.data.mul_(mask)
        bn_module.bias.data.mul_(mask)

    return losses_model


def gen_bn_channel_dict(model, ignore_bn_list, thresh):
    mask_bn_clannel_dict = dict()
    for bnname, bnlayer in model.named_modules():
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            mask = bn_module.weight.data.abs().ge(thresh).float()
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size())
            mask_bn_clannel_dict[bnname] = int(mask.sum())
    return mask_bn_clannel_dict


def init_weights_from_loose_model(loose_model, prune_mask_dict, prune_bn_dict, mask_bn_channel_dict):

    pruned_model = yolox_prun(mask_bn_channel_dict)
    prune_bn_keys = prune_bn_dict.keys()
    if isinstance(pruned_model.head,YOLOXHead_prune):
        convIn_from_ = {**pruned_model.backbone.get_convIn_from(),**pruned_model.head.get_convIn_from()}
    else:
        convIn_from_ = pruned_model.backbone.get_convIn_from()

    for (loose_name, loose_layer), (pruned_name, pruned_layer) in zip(loose_model.named_modules(), pruned_model.named_modules()):
        assert loose_name == pruned_name, print(loose_name, pruned_name)
        if isinstance(loose_layer, nn.BatchNorm2d):
            if loose_name in prune_bn_keys:
                out_idx = np.squeeze(np.argwhere(np.asarray(
                    prune_mask_dict[loose_name].cpu().numpy())))
                pruned_layer.weight.data = loose_layer.weight.data[out_idx].clone()
                pruned_layer.bias.data = loose_layer.bias.data[out_idx].clone()
                pruned_layer.running_mean = loose_layer.running_mean[out_idx].clone()
                pruned_layer.running_var = loose_layer.running_var[out_idx].clone()
            else:
                pruned_layer.weight.data = loose_layer.weight.data.clone()
                pruned_layer.bias.data = loose_layer.bias.data.clone()
                pruned_layer.running_mean = loose_layer.running_mean.clone()
                pruned_layer.running_var = loose_layer.running_var.clone()
        elif isinstance(loose_layer, nn.Conv2d):
            bn_name = loose_name[:-6] + ".bn" if loose_name[-6] == "." else loose_name[:-5] + ".bn"
            if bn_name in prune_bn_keys:
                convIn = convIn_from_[bn_name]
                if isinstance(convIn, int):
                    out_idx = np.squeeze(np.argwhere(np.asarray(
                        prune_mask_dict[bn_name].cpu().numpy())))
                    w = loose_layer.weight.data[out_idx, :, :, :].clone()
                    assert len(w.shape) == 4
                    pruned_layer.weight.data = w.clone()
                elif isinstance(convIn, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(
                        prune_mask_dict[bn_name].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(
                        prune_mask_dict[convIn].cpu().numpy())))
                    w = loose_layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) == 3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w.clone()
                elif isinstance(convIn, list):
                    out_idx = np.squeeze(np.argwhere(np.asarray(
                        prune_mask_dict[bn_name].cpu().numpy())))
                    in_idx = list()
                    for i in range(len(convIn)):
                        in_idx1 = np.squeeze(np.argwhere(np.asarray(prune_mask_dict[convIn[i]].cpu().numpy())))
                        in_idx += in_idx1.tolist()
                    w = loose_layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) == 3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w.clone()
            else:
                # if "head.cls_preds" in loose_name:
                #     i = loose_name.split(".")[-1]
                #     in_idx = np.squeeze(np.argwhere(np.asarray(
                #         prune_mask_dict["head.cls_convs.{}.1.bn".format(i)].cpu().numpy())))
                #     w = loose_layer.weight.data[:, in_idx, :, :].clone()
                #     if len(w.shape) == 3:
                #         w = w.unsqueeze(0)
                #     pruned_layer.weight.data = w.clone()
                # elif "head.reg_preds" in loose_name or \
                #     "head.obj_preds" in loose_name:
                #     i = loose_name.split(".")[-1]
                #     in_idx = np.squeeze(np.argwhere(np.asarray(
                #         prune_mask_dict["head.reg_convs.{}.1.bn".format(i)].cpu().numpy())))
                #     w = loose_layer.weight.data[:, in_idx, :, :].clone()
                #     if len(w.shape) == 3:
                #         w = w.unsqueeze(0)
                #     pruned_layer.weight.data = w.clone()
                if "stem" in loose_name:
                    a = ["backbone.C3_p3.conv3.bn", "backbone.C3_n3.conv3.bn", "backbone.C3_n4.conv3.bn"]
                    i = loose_name.split(".")[-2]
                    in_idx = np.squeeze(np.argwhere(np.asarray(
                        prune_mask_dict[a[int(i)]].cpu().numpy())))
                    w = loose_layer.weight.data[:, in_idx, :, :].clone()
                    if len(w.shape) == 3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w.clone()
                else:
                    pruned_layer.weight.data = loose_layer.weight.data.clone()
        else:
            pruned_layer = loose_layer
    return pruned_model


def main(exp,args):
    model = exp.get_model()
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    t_start = time.time()
    compact_model_out = model(dummy_input)
    print(time.time() - t_start)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model"])

    def obtain_num_parameters(model): return sum([param.nelement() for param in model.parameters()])
    origin_nparameters = obtain_num_parameters(model)
    print(origin_nparameters)

    prune_bn_dict, ignore_bn_list = gen_bn_module(model)
    prune_bn_weights = gather_bn_weights(prune_bn_dict)

    sorted_bn = torch.sort(prune_bn_weights)[0]
    sorted_bn, sorted_index = torch.sort(prune_bn_weights)
    thresh_index = int(len(prune_bn_weights) * 0.2)
    thresh = sorted_bn[thresh_index]

    prune_ratio, prune_mask_dict = obtain_filters_mask(prune_bn_dict, thresh)

    loose_model = prune_model_keep_size(model, prune_mask_dict)

    mask_bn_channel_dict = gen_bn_channel_dict(loose_model, ignore_bn_list, thresh)

    pruned_model = init_weights_from_loose_model(
        loose_model, prune_mask_dict, prune_bn_dict, mask_bn_channel_dict)

    pruned_model.eval()

    dummy_input = torch.randn(1, 3, 640, 640)

    t_start = time.time()
    compact_model_out = pruned_model(dummy_input)
    print(time.time() - t_start)
    print(obtain_num_parameters(pruned_model))
    

    trainer = MY_Trainer_Fine(exp, args)
    trainer.train(pruned_model)

    torch.onnx._export(
        pruned_model,
        dummy_input,
        "yolox_pruned.onnx",
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: 'batch'},
                      "output": {0: 'batch'}} if False else None,
        opset_version=11,
    )
    ckpt_state = {
        "model": pruned_model.state_dict(),
    }
    torch.save(ckpt_state, "pruned.pth")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp,args)
