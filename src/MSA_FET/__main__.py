import argparse

from .main import FeatureExtractionTool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input video file in file mode, or dataset root directory in dataset mode.")
    parser.add_argument('-d', '--dataset-mode', action='store_true',
                        help="Switch from file mode to dataset mode if specified.")
    parser.add_argument('-c', '--config-file', type=str, required=True,
                        help="Path to config file.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to output pkl file.")
    parser.add_argument('-t', '--text-file', type=str, required=False,
                        help="File containing transcriptions of the video in file mode.")
    parser.add_argument('-n', '--num-workers', type=int, default=4,
                        help="Number of workers for data loading in dataset mode.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more information to stdout.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Print only errors to stdout.")
    return parser.parse_args()


if __name__ == '__main__':
    # TODO: add more specific arguments, such as output format
    args = parse_args()

    verbose = 1
    if args.verbose:
        verbose = 2
    if args.quiet:
        verbose = 0
    
    fet = FeatureExtractionTool(config=args.config_file, verbose=verbose)
    if args.dataset_mode:
        fet.run_dataset(
            dataset_dir=args.input,
            out_file=args.output,
            num_workers=args.num_workers
        )
    else:
        fet.run_single(
            in_file=args.input,
            out_file=args.output,
            text_file=args.text_file
        )
