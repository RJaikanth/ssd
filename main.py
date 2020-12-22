import os
import json
import time
import argparse
import warnings

from src.train import trainAndEval
from src.data_lists import create_data_lists
from src.utils.helpers import read_config, check_config


warnings.filterwarnings("ignore", category=UserWarning)

DESCRIPTION = """This is the main script that calls the other functions"""
CONFIG_PATH = os.path.abspath("./config/")

# Help Strings
HELP_DICT = {
    "data_lists": "Set flag if you want to create data lists",
    "train": "Set flag if you want to train the model.",
    "data_config": "Path to Data Configuration file. Should be in json format.",
    "model_config": "Path to Model Configuration file. Should be in json format.",
    "exp_config": "Path to Experiment Configuration file. Should be in json format.",
}


args = argparse.ArgumentParser(description=DESCRIPTION)
args.add_argument("--data_lists", default=False, action="store_true", help=HELP_DICT["data_lists"])
args.add_argument("--train", default=False, action="store_true", help=HELP_DICT["train"])
args.add_argument("--data_config", default="data_config.json", help=HELP_DICT["data_config"], metavar="")
args.add_argument("--model_config", default="model_config.json", help=HELP_DICT["model_config"], metavar="")
args.add_argument("--exp_config", help=HELP_DICT["exp_config"], metavar="")


if __name__ == "__main__":
    print("\n" + "==" * 50)
    print("Script Started at - {}".format(time.strftime("%d-%B-%Y %H:%M:%S")))
    print("Reading and parsing arguments")
    args = args.parse_args()
    print("Parsed arguments")
    print("\n" + "==" * 50)

    # DATA_LISTS CALL
    if args.data_lists is True:
        print("Function Called: Create Data lists")

        # Check and read config.
        check_config(args, "data_config")
        config = read_config(os.environ["CONFIG_DIR"], args.data_config)

        # Call function
        create_data_lists(config)

    # MODEL_TRAINING CALL
    if args.train is True:
        print("Function Called: Train Model")

        # Check configs.
        check_config(args, "exp_config")
        check_config(args, "data_config")
        check_config(args, "model_config")

        # Read config.
        exp_config = read_config(os.environ["CONFIG_DIR"], args.exp_config)
        data_config = read_config(os.environ["CONFIG_DIR"], args.data_config)
        model_config = read_config(os.environ["CONFIG_DIR"], args.model_config)

        # Call main function
        trainAndEval(data_config, model_config, exp_config)

    print("\n" + "==" * 50)
    print("Script completed succesfully at {}".format(time.strftime("%d-%B-%Y %H:%M:%S")))
    print("\n" + "==" * 50 + "\n")
