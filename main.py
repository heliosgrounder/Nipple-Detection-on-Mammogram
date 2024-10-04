import argparse

from src.trainer import Trainer
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Nipple Detection")
    parser.add_argument(
        "--mode",
        help="Mode of program: train or run",
        default="run",
        required=True
    )
    parser.add_argument(
        "--root_dir", 
        help="Path to dataset directory",
        default="data", 
        required=False
    )
    parser.add_argument(
        "--presentationLUT",
        help="presentationLUT attribute IDENTITY|INVERSE",
        default="IDENTITY",
        required=False
    )
    parser.add_argument(
        "--holdout_dir",
        help="Path to a holdout set directory",
        default="data_check",
        required=False
    )
    parser.add_argument(
        "--verbose",
        help="Print debug information",
        default=True,
        required=False
    )
    parser.add_argument(
        "--imshow",
        help="Show image with detected nipples",
        default=True,
        required=False
    )
    parser.add_argument(
        "--delete_check_folders",
        help="Delete check folders",
        default=False,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.mode == "train":
        trainer = Trainer(
            presentationLUT=args.presentationLUT,
            verbose=args.verbose,
            epochs=1000
        )
        trainer.train()
    else:
        runner = Runner(
            folder_check=args.holdout_dir,
            presentationLUT=args.presentationLUT,
            verbose=args.verbose,
            imshow=args.imshow
        )
        runner.load_model("models")
        runner.run()
        if args.delete_check_folders:
            runner.delete_check_folders()