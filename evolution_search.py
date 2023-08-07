'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time, yaml
import torch
from torch import nn
import numpy as np
import global_utils
from pathlib import Path
from copy import deepcopy
from utils.general import check_anchor_order, make_divisible, check_file
# from tqdm import tqdm

from models.yolo import parse_model, Model, get_FLOP, get_score, get_layers
from ZeroShotProxy import compute_tpc_score
from ptflops import get_model_complexity_info
# import benchmark_network_latency

working_dir = os.path.dirname(os.path.abspath(__file__))

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(48e4),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes')
    parser.add_argument('--yaml_file', type=str, default=None,
                        help='input yaml file')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

def get_new_random_structure_str(yaml_file, mutate_count=5):
    # the_net, save = parse_model(deepcopy(yaml_file), ch=[3])
    def sub_layers_mutator(sub_layers):
        the_list = [sub_layers,
                sub_layers + 1, sub_layers + 2,
                sub_layers - 1, sub_layers - 2]

        the_sub_layers = random.choice(the_list)
        the_sub_layers = min(max(the_sub_layers, 2), 10)

        return the_sub_layers

    def channel_mutator(out_channels, no):
        # print(out_channels)
        the_list = [out_channels * 2.5, out_channels * 2, out_channels * 1.5, out_channels * 1.25,
                out_channels,
                out_channels / 1.25, out_channels / 1.5, out_channels / 2, out_channels / 2.5]

        the_out_channels = random.choice(the_list)
        out_channels = make_divisible(the_out_channels, 8) if out_channels != no else out_channels

        return out_channels


    d = deepcopy(yaml_file)
    ch = [3]
    # print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # check all the layer to mutate
    mutate_layer = []
    # layer_count = len(d['backbone'] + d['head'])
    layer_count = len(d['backbone'])
    for i in range(mutate_count):
        the_layer = random.randint(0, layer_count-1)
        while the_layer in mutate_layer:
            the_layer = random.randint(0, layer_count-1)
        mutate_layer.append(the_layer)

    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        if i in mutate_layer:

            # sub_layers count mutator
            if(n>1):
                sub_layers = sub_layers_mutator(n)
                d['backbone'][i][1] = sub_layers

            out_channel = args[0]
            if isinstance(out_channel, int) and out_channel is not None and m != "Concat":
                d['backbone'][i][3][0] = channel_mutator(out_channel, no)

            if m == "Conv":
                if args[1] > 1 :
                    the_kernel_list = [3, 5, 7]
                    the_kernel_size = random.choice(the_kernel_list)
                    d['backbone'][i][3][1] = the_kernel_size
        else:
            continue

    for i, (f, n, m, args) in enumerate(d['head']):  # from, number, module, args
        if (i+len(d['head'])) in mutate_layer:

            # sub_layers count mutator
            if(n>1):
                sub_layers = sub_layers_mutator(n)
                d['head'][i][1] = sub_layers

            out_channel = args[0]
            if isinstance(out_channel, int) and out_channel is not None and m != "Concat":
                d['head'][i][3][0] = channel_mutator(out_channel, no)

            if m == "Conv":
                if args[1] > 1 :
                    the_kernel_list = [3, 5, 7]
                    the_kernel_size = random.choice(the_kernel_list)
                    d['head'][i][3][1] = the_kernel_size
        else:
            continue

    # print(d)
    return d


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str

# def get_latency(AnyPlainNet, random_structure_str, gpu, args):
#     the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
#                             no_create=False, no_reslink=False)
#     if gpu is not None:
#         the_model = the_model.cuda(gpu)
#     the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
#                                                               resolution=args.input_image_size,
#                                                               in_channels=3, gpu=gpu, repeat_times=1,
#                                                               fp16=True)
#     del the_model
#     torch.cuda.empty_cache()
#     return the_latency

def compute_nas_score(the_model, random_structure_str, gpu, args):
    # compute network zero-shot proxy score
    # the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,no_create=False, no_reslink=True)
    # the_model = the_model.cuda(gpu)
    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TPC':
            # print("hello")
            the_nas_core_info = compute_tpc_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)

        elif args.zero_shot_score == 'Syncflow':
            the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                       resolution=args.input_image_size,
                                                                       batch_size=args.batch_size)

        elif args.zero_shot_score == 'GradNorm':
            the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)

        elif args.zero_shot_score == 'Flops':
            the_nas_core = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_core = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_core = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
    except Exception as err:
        logging.info(str(err))
        logging.info('--- Failed structure: ')
        # logging.info(str(the_model))
        # raise err
        the_nas_core = -9999


    del the_model
    torch.cuda.empty_cache()
    return the_nas_core

def main(args, argv):
    gpu = args.gpu
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.yaml')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # load search space config .py file
    yaml_file = Path(args.yaml_file).name
    with open(args.yaml_file) as f:
        yaml_model = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    # model = Model(args.yaml_file)
    # the_model_flops, the_model_size = get_model_complexity_info(model, (3, 640, 640), as_strings=False,
    #                                        print_per_layer_stat=False, verbose=False)
    # my_flops = get_FLOP(yaml_model, 640)

    # the_nas_core = compute_nas_score(model, yaml_model, gpu, args)
    # my_score = get_score(yaml_model)
    # my_layer = get_layers(yaml_model)
    # print("initial FLOPs = ", the_model_flops)
    # print("my flops  = ", my_flops)
    # print("initial size  = ", the_model_size)

    # print("original score = ", the_nas_core)
    # print("my score  = ", my_score)

    # print("layer_count = ", my_layer)

    popu_structure_list         = []
    popu_zero_shot_score_list   = []
    popu_latency_list           = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter):
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # start_time = time.time()
        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(yaml_file=yaml_model,mutate_count=5)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(yaml_file=tmp_random_structure_str,mutate_count=2)

        if args.max_layers is not None:
            the_layers = get_layers(random_structure_str)
            if args.max_layers < the_layers:
                continue

        the_model_flops = get_FLOP(random_structure_str, args.input_image_size)
        # print("phase 3 = ", time.time()-start_time)


        if args.budget_model_size is not None:
            # if the_model is None:
            #     the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
            #                             no_create=True, no_reslink=False)
            # the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            # if the_model is None:
            #     the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
            #                             no_create=True, no_reslink=False)
            # the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        # if args.budget_latency is not None:
        #     the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
        #     if args.budget_latency < the_latency:
        #         continue

        the_nas_core = get_score(random_structure_str)

        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list

if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.yaml')
    global_utils.mkfilepath(best_structure_txt)
    # with open(best_structure_txt, 'w') as fid:
    #     fid.write(best_structure_str)
    # pass  # end with
    with open(best_structure_txt, "w") as f:
        yaml.dump(best_structure_str, f)
