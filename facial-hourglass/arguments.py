import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General
    parser.add_argument(
        '--img_glob', default='./datasets/IBUG_300_W/*/*.png', type=str, help='Image paths')
    parser.add_argument(
        '--pts_glob', default='./datasets/IBUG_300_W/*/*.pts', type=str, help='Point paths')
    parser.add_argument(
        '--gpu_id', default='0', type=int, help='id of the gpu to use')

    # Hyperparams
    parser.add_argument(
        '--dataset_ratio', default='0.8', type=int, help='Ratio to split train and test. Default is 0.8 so train/test = 0.8/0.2')
    parser.add_argument(
        '--batchsize', default='32', type=int, help='batchsize')
    parser.add_argument(
        '--max_epoch', default='20', type=int, help='Number of epochs to train')
    parser.add_argument(
        '--out', default='./results', type=str, help='Specifies an output directory used to save the log files')


    parser.add_argument(
        '--pts_mode', default='heatmap', type=str, help='return keypoints as heatmaps or points')
