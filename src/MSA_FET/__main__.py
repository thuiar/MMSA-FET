import argparse
from pathlib import Path

from .main import FeatureExtractionTool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input video file in file mode, or dataset dir in dataset mode.")
    parser.add_argument('-d', '--dataset-mode', action='store_true',
                        help="Switch from file mode to dataset mode if specified.")
    parser.add_argument('-c', '--config-file', type=str, required=True,
                        help="Path to config file.")
    parser.add_argument('-o', '--output', type=str, required=True, # can only write to file in commandline
                        help="Path to output pkl file.")
    parser.add_argument('-t', '--text-file', type=str, default=None,
                        help="File containing transcriptions of the video. Required for extracting text features for single video.")
    parser.add_argument('-n', '--num-workers', type=int, default=4,
                        help="Number of workers for data loading in dataset mode.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more information to stdout.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Print only errors to stdout.")
    parser.add_argument('--return_type', type=str, default='np', choices=['np', 'pt', 'df'],
                        help="Return type of the tool.")
    parser.add_argument('--tmp-dir', type=str, default=Path.home() / '.MMSA-FET/tmp',
                        help="Temporary directory for storing intermediate results. Default: '~/.MSA-FET/tmp'")
    parser.add_argument('--log-dir', type=str, default=Path.home() / '.MMSA-FET/log',
                        help="Log file directory. Default: '~/.MSA-FET/log'")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch size for dataset mode. Default: 4")
    parser.add_argument()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    verbose = 1
    if args.verbose:
        verbose = 2
    if args.quiet:
        verbose = 0
    
    # if args.tmp_dir == '':
    #     args.tmp_dir = Path.home() / '.MMSA-FET/tmp'
    # if args.log_dir == '':
    #     args.log_dir = Path.home() / '.MMSA-FET/log'
    
    # if args.text_file == '':
    #     args.text_file = None
    
    fet = FeatureExtractionTool(
        config=args.config_file,
        tmp_dir=args.tmp_dir,
        log_dir=args.log_dir,
        verbose=verbose
    )

    if args.dataset_mode:
        fet.run_dataset(
            dataset_dir=args.input,
            out_file=args.output,
            num_workers=args.num_workers,
            return_type=args.return_type,
            batch_size=args.batch_size
        )
    else:
        fet.run_single(
            in_file=args.input,
            out_file=args.output,
            text_file=args.text_file,
            return_type=args.return_type
        )
