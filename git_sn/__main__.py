
from git_sn.cli import get_parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)
