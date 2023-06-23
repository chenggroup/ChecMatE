import sys, argparse
from .workflow import wf_init_data, wf_explore, wf_gen_mlps


def parse_args():
    """
    autowf commandline options argument parser.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command",
                        help="workflow type",
                        choices=["init_data", "gen_mlps", "explore"])
    
    parser.add_argument("-s",
                        "--setting",
                        help="user setting file with .yaml or .json format",
                        required=True,
                        type=str)
    
    parser.add_argument("-m",
                        "--max_task",
                        help="(command: gen_mlps) the maximum number of task in a workflow",
                        type=int,
                        default=300)

    parser.add_argument("-l",
                        "--long_train",
                        help="(command: gen_mlps) the maximum number of task in a workflow",
                        action="store_true",
                        default=False)
    
    parser.add_argument("-wo",
                        "--without",
                        help="(command: init_data and explore) \
                            without sampling process in init_data workflow \
                            or without labeling process in explore workflow",
                        action="store_true",
                        default=False)
    
    parsed_args = parser.parse_args()
    if parsed_args.command is None:
        parser.print_help()
    return parsed_args


def gen_mlps(**kwargs):
    setting = kwargs["setting"]
    max_task = kwargs["max_task"]
    long_train = kwargs["long_train"]
    proj = wf_gen_mlps
    proj.gen_mlps_flow(setting, max_task, whether_long_train=long_train)


def init_data(**kwargs):
    setting = kwargs["setting"]
    whether_to_sample = not(kwargs["without"])
    proj = wf_init_data
    if whether_to_sample:
        proj.dataset_sswfp_flow(setting)
    else:
        proj.dataset_fp_flow(setting)


def explore(**kwargs):
    setting = kwargs["setting"]
    whether_to_label = not(kwargs["without"])
    proj = wf_explore
    if whether_to_label:
        proj.explore_flow(setting)
    else:
        proj.explore_flow_without_fp(setting)


def main():
    args = parse_args()
    kwargs = vars(args)
    if args.command in ["init_data", "gen_mlps", "explore"]:
        getattr(sys.modules[__name__], args.command)(**kwargs)
    else:
        raise RuntimeError(f"unknown command {args.command}")
