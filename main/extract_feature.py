# -*- coding: utf-8 -*-

import argparse
import os

import torch
import sys
from pathlib import Path
sys.path.append(r"D:\Program\Pycharm\pythonProject\PyRetri-master")
from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.datasets import build_folder, build_loader
from pyretri.models import build_model
from pyretri.extract import build_extract_helper

from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--data_json', '-dj', default=None, type=str, help='json file for dataset to be extracted')
    parser.add_argument('--save_path', '-sp', default=None, type=str, help='save path for features')
    parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--save_interval', '-si', default=5000, type=int, help='number of features saved in one part file')
    args = parser.parse_args()
    return args


def main(args):

    # init args
    # args = parse_args()
    # print(f"{args.data_json}\n{args.save_path}\n{args.config_file}\n{args.opts}\n{args.save_interval}")
    assert args.data_json is not None, 'the dataset json must be provided!'
    assert args.save_path is not None, 'the save path must be provided!'
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()   # 应该是设置一个初始文件树 默认值之类的，只是不明白为什么里面要牵涉到那么多文件
    cfg = setup_cfg(cfg, args.config_file, args.opts)  # 上面应该是初始话cfg这个文件，然后将yaml设置的值存入初始化的文件中

    # build dataset and dataloader
    dataset = build_folder(args.data_json, cfg.datasets)
    dataloader = build_loader(dataset, cfg.datasets)

    # build model
    model = build_model(cfg.model)

    # build helper and extract features
    extract_helper = build_extract_helper(model, cfg.extract)
    extract_helper.do_extract(dataloader, args.save_path, args.save_interval)


if __name__ == '__main__':
    # args = argparse.Namespace(data_json=r"C:\Users\datateam001\Desktop\New_folder\data_json\caltech_gallery.json",
    #                           save_path= r"C:\Users\datateam001\Desktop\New_folder\feature\caltech\gallery",
    #                           config_file=r"D:\Program\Pycharm\pythonProject\PyRetri-master\configs\caltech.yaml",
    #                           opts=argparse.REMAINDER,
    #                           save_interval=5000)
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--data_json', '-dj', default=r"C:\Users\datateam001\Desktop\New folder\data_json\caltech_gallery.json", type=str, help='json file for dataset to be extracted')
    parser.add_argument('--save_path', '-sp', default=r"C:\Users\datateam001\Desktop\New folder\feature\caltech\gallery", type=str, help='save path for features')
    parser.add_argument('--config_file', '-cfg', default=r"D:\Program\Pycharm\pythonProject\PyRetri-master\configs\caltech.yaml", metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--save_interval', '-si', default=5000, type=int, help='number of features saved in one part file')
    args = parser.parse_args()

    # args = parse_args()
    main(args)


