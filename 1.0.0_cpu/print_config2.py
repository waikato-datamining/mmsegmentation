# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) University of Waikato, Hamilton, NZ
import argparse

from mmengine import Config, DictAction

from mmseg.apis import init_model


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('--config', help='config file path')
    parser.add_argument(
        '--graph', action='store_true', help='print the models graph')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--output_config', required=True, help='where to store the generated config')
    parser.add_argument(
        '--output_graph', required=False, help='where to store the graph (requires --graph)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(f'Config:\n{cfg.pretty_text}')
    # dump config
    cfg.dump(args.output_config)
    # dump models graph
    if args.graph:
        model = init_model(args.config, device='cpu')
        print(f'Model graph:\n{str(model)}')
        with open(args.output_graph, 'w') as f:
            f.writelines(str(model))


if __name__ == '__main__':
    main()
