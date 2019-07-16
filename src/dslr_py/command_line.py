import argparse
import logging
import os
import sys

from sklearn.utils.murmurhash import murmurhash3_32

from dslr_py import utils
from dslr_py.dslr import Processor

__all__ = ["main"]


def main(fp=sys.stdout, argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='DSLR')
    parser.add_argument('source', type=str)
    parser.add_argument('destination', type=str)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--log', type=str, default='INFO')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--partition', type=str, default='0/1')
    parser.add_argument('--config', type=str)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    partition_no, partitions_count = args.partition.split('/')
    partition_no = int(partition_no)
    partitions_count = int(partitions_count)
    assert partitions_count > 0

    def _f(f):
        if partition_no >= 0:
            file_partition = murmurhash3_32(f, args.seed) % partitions_count
            if file_partition != partition_no:
                return False
        file_name = f.rsplit(".", 1)[0]
        file_path = dest_dir + 'large' + '/v1_' + file_name + '.jpg'
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return False
        return True

    source_dir = os.path.realpath(os.path.expanduser(args.source.rstrip('/'))) + '/'
    dest_dir = os.path.realpath(os.path.expanduser(args.destination.rstrip('/'))) + '/'

    files = [f for f in os.listdir(source_dir) if os.path.isfile(source_dir + f) and f.rsplit(".", 1)[1] == 'jpg']
    files = list(filter(_f, files))

    if args.limit is not None:
        files = files[:args.limit]

    p = Processor(
        map(lambda x: [source_dir, x, dest_dir], files),
        len(files),
        utils.load_config(args.config),
        seed=args.seed, device=args.device
    )
    log = logging.getLogger('clt')
    log.info('Starting')

    for i in p.start():
        pass
