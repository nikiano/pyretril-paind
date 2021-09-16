# -*- coding: utf-8 -*-

import argparse
import os
import pickle


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    # parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--config_file', '-cfg', default=r"D:\Program\Pycharm\pythonProject\PyRetri-master\configs\caltech.yaml", metavar='FILE', type=str, help='path to config file')
    args = parser.parse_args()
    return args


def main():

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings # 这两行，解压初始化参数
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    # load features # 这两行，将训练后提取的信息，读取到内存中
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    # build helper and index features # 这里是数据预处理。这里就是建立类的索引，然后把数据放到个各种类里面，输出，查询的标签，查询特征，图片库特征
    index_helper = build_index_helper(cfg.index) # 这里放的都是模型的索引，各种类的集合体
    index_result_info, query_fea, gallery_fea = index_helper.do_index(query_fea, query_info, gallery_fea)

    # build helper and evaluate results
    evaluate_helper = build_evaluate_helper(cfg.evaluate)
    mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_info)

    # show results
    evaluate_helper.show_results(mAP, recall_at_k)


if __name__ == '__main__':
    main()
