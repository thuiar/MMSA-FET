import argparse
from pathlib import Path

from .main import FeatureExtractionTool
from .install import download_missing, force_redownload


def parse_args():
    # main parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print more information to stdout.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Print only errors to stdout.")
    parser.add_argument('--tmp-dir', type=str, default=Path.home() / '.MMSA-FET/tmp',
                        help="Temporary directory for storing intermediate results. Default: '~/.MSA-FET/tmp'")
    parser.add_argument('--log-dir', type=str, default=Path.home() / '.MMSA-FET/log',
                        help="Log file directory. Default: '~/.MSA-FET/log'")

    # sub-parser for installation
    parser_install = subparsers.add_parser("install", help="Download models required to run MSA-FET")
    parser_install.add_argument('-p', '--proxy', type=str, default=None,
                                help="Proxy for downloading. e.g. socks5://127.0.0.1:8080")
    parser_install.add_argument('-f', '--force', action='store_true', help="Force re-download models.")
    
    # sub-parser for dataset mode
    parser_single = subparsers.add_parser("run_single", help="Run feature extraction on a single video")
    parser_single.add_argument('-i', '--input', type=str, required=True,
                                help="Input video file in file mode, or dataset dir in dataset mode.")
    parser_single.add_argument('-c', '--config-file', type=str, required=True,
                                help="Path to config file.")
    parser_single.add_argument('-o', '--output', type=str, required=True,
                                help="Path to output pkl file.")
    parser_single.add_argument('-t', '--text-file', type=str, default=None,
                                help="File containing transcriptions of the video. Required when extracting text features.")
    parser_single.add_argument('--return_type', type=str, default='np', choices=['np', 'pt', 'df'],
                                help="Return type of the tool.")
    

    parser_dataset = subparsers.add_parser("run_dataset", help="Run feature extraction on a dataset")
    parser_dataset.add_argument('-i', '--input', type=str, required=True,
                                help="Input dataset dir.")
    parser_dataset.add_argument('-c', '--config-file', type=str, required=True,
                                help="Path to config file.")
    parser_dataset.add_argument('-o', '--output', type=str, required=True,
                                help="Path to output pkl file.")
    parser_dataset.add_argument('-n', '--num-workers', type=int, default=4,
                                help="Num of dataloader workers.")
    parser_dataset.add_argument('--batch-size', type=int, default=4,
                                help="Batch size. Default: 4")
    parser_dataset.add_argument('--return_type', type=str, default='np', choices=['np', 'pt', 'df'],
                                help="Return type of the tool.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'install':
        if args.force:
            force_redownload(args.proxy)
        else:
            download_missing(args.proxy)
    elif args.command is None:
        print("Please specify a command. Use -h to see all available commands.")
    else:
        verbose = 1
        if args.verbose:
            verbose = 2
        if args.quiet:
            verbose = 0
        
        fet = FeatureExtractionTool(
            config=args.config_file,
            tmp_dir=args.tmp_dir,
            log_dir=args.log_dir,
            verbose=verbose
        )

        if args.command == 'run_single':
            fet.run_single(
                in_file=args.input,
                out_file=args.output,
                text_file=args.text_file,
                return_type=args.return_type
            )
        elif args.command == 'run_dataset':
            fet.run_dataset(
                dataset_dir=args.input,
                out_file=args.output,
                num_workers=args.num_workers,
                return_type=args.return_type,
                batch_size=args.batch_size
            )
        else:
            print("Unknown command: {}".format(args.command))
