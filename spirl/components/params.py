import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")

    # Folder settings
    parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory")
    parser.add_argument('--new_dir', default=False, type=int, help='If True, concat datetime string to exp_dir.')
    parser.add_argument('--dont_save', default=False, type=int,
                        help="if True, nothing is saved to disk. Note: this doesn't work")  # TODO this doesn't work

    # Running protocol
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--train', default=True, type=int,
                        help='if False, will run one validation epoch')
    parser.add_argument('--test_prediction', default=True, type=int,
                        help="if False, prediction isn't run at validation time")
    parser.add_argument('--skip_first_val', default=False, type=int,
                        help='if True, will skip the first validation epoch')
    parser.add_argument('--val_sweep', default=False, type=int,
                        help='if True, runs validation on all existing model checkpoints')

    # Misc
    parser.add_argument('--gpu', default=-1, type=int,
                        help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--strict_weight_loading', default=True, type=int,
                        help='if True, uses strict weight loading function')
    parser.add_argument('--deterministic', default=False, type=int,
                        help='if True, sets fixed seeds for torch and numpy')
    parser.add_argument('--log_interval', default=500, type=int,
                        help='number of updates per training log')
    parser.add_argument('--per_epoch_img_logs', default=1, type=int,
                        help='number of image loggings per epoch')
    parser.add_argument('--val_data_size', default=-1, type=int,
                        help='number of sequences in the validation set. If -1, the full dataset is used')
    parser.add_argument('--val_interval', default=5, type=int,
                        help='number of epochs per validation')

    # Debug
    parser.add_argument('--detect_anomaly', default=False, type=int,
                        help='if True, uses autograd.detect_anomaly()')
    parser.add_argument('--feed_random_data', default=False, type=int,
                        help='if True, we feed random data to the model to test its performance')
    parser.add_argument('--train_loop_pdb', default=False, type=int,
                        help='if True, opens a pdb into training loop')
    parser.add_argument('--debug', default=False, type=int,
                        help='if True, runs in debug mode')

    # add kl_div_weight
    parser.add_argument('--save2mp4', default=False, type=bool,
                        help='if set, videos will be saved locally')

    return parser.parse_args()
