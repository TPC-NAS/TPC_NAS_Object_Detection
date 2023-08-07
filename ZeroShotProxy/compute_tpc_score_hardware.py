import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time
from PlainNet import basic_blocks
import math

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(random_structure_str):
    # print(random_structure_str)
    # model.eval()
    info = {}
    nas_score_list = []

    assert len(random_structure_str) % 32 == 0
    block_count = len(random_structure_str)//32

    test_log_conv_scaling_factor = 0

    for i in range(block_count):
        the_block = random_structure_str[32*i:32*(i+1)]
        block_type    = the_block[0:3]
        in_channel    = int(the_block[3:11], 2)  * 8
        out_channel   = int(the_block[11:19], 2) * 8
        stride        = int(the_block[19:21], 2)
        bottleneck    = int(the_block[21:29], 2) * 8
        sublayers     = int(the_block[29:32], 2)

        if block_type == "010" or block_type == "101" or block_type == "000":
            kernel_size = 3
        elif block_type == "001":
            kernel_size = 1
        elif block_type == "011" or block_type == "110":
            kernel_size = 5
        elif block_type == "100" or block_type == "111":
            kernel_size = 7

        if block_type == "000":
            score = out_channel * (kernel_size**2) / 1.0
        elif block_type == "001":
            score = out_channel * (kernel_size**2) / 1.0
        elif block_type == "010" or block_type == "011" or block_type == "100":
            score = (((bottleneck**4) * (out_channel**2) * (kernel_size**4))**sublayers) / (stride**2)
        elif block_type == "101" or block_type == "110" or block_type == "111":
            score = ((bottleneck*out_channel*(kernel_size**4))**sublayers) / (stride**2)

        # print(score)
        # print()
        test_log_conv_scaling_factor += math.log(score)

        # print(test_log_conv_scaling_factor)


    # for name, module in model.named_modules():
    #     if isinstance(module, basic_blocks.ConvKX) or isinstance(module, basic_blocks.ConvDW):
    #         score = torch.tensor(float(module.out_channels * (module.kernel_size ** 2) // (module.stride **2)))
    #         print(score)
    #         test_log_conv_scaling_factor += torch.log(score)

    # nas_score = torch.tensor(1.0)
    nas_score = test_log_conv_scaling_factor
    nas_score_list.append(float(nas_score))

    assert not (nas_score != nas_score)

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)

    # print("avg_nas_score = ", avg_nas_score)

    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
