"""
Deletes all non-latest checkpoints in all subdirectories recursively to save memory.
WARNING: only do this if you do not want to resume any of your past model / RL training runs from an intermediate checkpoint!
"""

import os
import glob
import sys
import tqdm
from spirl.components.checkpointer import CheckpointHandler


def find_weight_folders():
    dirnames = []
    for root, dirs, files in os.walk(os.getcwd(), followlinks=True):
        for dir in dirs:
            if dir.endswith("weights"): dirnames.append(os.path.join(root, dir))
    return dirnames


def delete_non_latest_checkpoint(dir):
    latest_checkpoint = CheckpointHandler.get_resume_ckpt_file("latest", dir)
    checkpoint_names = glob.glob(os.path.abspath(dir) + "/*.pth")
    for file in checkpoint_names:
        if file != latest_checkpoint:
            os.remove(file)


def query_yes_no(question, default=None):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    Copied from here: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "no": False}
    if default is None:
        prompt = " [yes/no] "
    elif default == "yes":
        prompt = " [YES/no] "
    elif default == "no":
        prompt = " [yes/NO] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")


if __name__ == "__main__":
    if not query_yes_no("Deleting all non-latest checkpoints from {}, CONTINUE?".format(os.getcwd())):
        print("Aborting...")
        exit(0)

    checkpt_dirs = find_weight_folders()

    if not query_yes_no("Will delete checkpoints from {} directories, CONTINUE?".format(len(checkpt_dirs))):
        print("Aborting...")
        exit(0)

    for checkpt_dir in tqdm.tqdm(checkpt_dirs):
        delete_non_latest_checkpoint(checkpt_dir)

    print("Done!")
