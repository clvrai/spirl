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
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'val', 'rollout'],
                        help='mode of the program (training, validation, or generate rollout)')

    # Misc
    parser.add_argument('--seed', default=-1, type=int,
                        help='overrides config/default seed for more convenient seed setting.')
    parser.add_argument('--gpu', default=-1, type=int,
                        help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--strict_weight_loading', default=True, type=int,
                        help='if True, uses strict weight loading function')
    parser.add_argument('--deterministic', default=False, type=int,
                        help='if True, sets fixed seeds for torch and numpy')
    parser.add_argument('--n_val_samples', default=10, type=int,
                        help='number of validation episodes')
    parser.add_argument('--save_dir', type=str,
                        help='directory for saving the generated rollouts in rollout mode')
    parser.add_argument('--config_override', default='', type=str,
                        help='override to config file in format "key1.key2=val1,key3=val2"')

    # Debug
    parser.add_argument('--debug', default=False, type=int,
                        help='if True, runs in debug mode')

    # Note
    parser.add_argument('--notes', default='', type=str,
                        help='Notes for the run')

    return parser.parse_args()
