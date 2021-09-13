"""
Deletes all stored replay buffers in all subdirectories recursively to save memory.
WARNING: only do this if you do not want to resume any of your past RL training runs!
"""

import os
import glob
import sys
import tqdm
from spirl.components.checkpointer import CheckpointHandler


def find_replay_files():
    filenames = []
    for root, dirs, files in os.walk(os.getcwd(), followlinks=True):
        for file in files:
            if file.endswith("replay_buffer.zip"): filenames.append(os.path.join(root, file))
    return filenames


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
    if not query_yes_no("Deleting all stored replay buffers from {}, CONTINUE?".format(os.getcwd())):
        print("Aborting...")
        exit(0)

    replay_files = find_replay_files()

    if not query_yes_no("Will delete {} replay buffer files, CONTINUE?".format(len(replay_files))):
        print("Aborting...")
        exit(0)

    for replay_file in tqdm.tqdm(replay_files):
        os.remove(replay_file)

    print("Done!")
