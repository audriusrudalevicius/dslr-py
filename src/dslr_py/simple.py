import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
from scipy import misc
from sklearn.utils.murmurhash import murmurhash3_32
from tqdm import tqdm

from dslr_py import image_size
from dslr_py.app_logging import configure_logging

__all__ = ['main']


def save_img_array(img_arr, file_path, quality=70):
    pil_image = misc.toimage(img_arr, channel_axis=2, cmin=0, cmax=255)
    pil_image.save(file_path, format='JPEG', quality=quality)


def main(fp=sys.stdout, argv=None):
    if argv is None:
        argv = sys.argv[1:]

    model_type = 'iphone_orig'
    # CUDA_VISIBLE_DEVICES="1"
    parser = argparse.ArgumentParser(description='DSLR')
    parser.add_argument('source', type=str)
    parser.add_argument('destination', type=str)
    parser.add_argument('--device', type=str, default='/gpu:0')
    parser.add_argument('--log', type=str, default='INFO')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--partition', type=str, default='0/1')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--version_name', type=str, default='v1')
    args = parser.parse_args(argv)

    configure_logging(args.log)
    log = logging.getLogger('app')

    partition_no, partitions_count = args.partition.split('/')
    partition_no = int(partition_no)
    partitions_count = int(partitions_count)

    def _f(f):
        if partition_no >= 0:
            file_partition = murmurhash3_32(f, args.seed) % partitions_count
            if file_partition != partition_no:
                return False
        file_name = f.rsplit(".", 1)[0]
        file_path = dest_dir + 'large' + '/v1_' + file_name + '.jpg'
        if args.no_skip:
            return True
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return False
        return True

    from dslr_py import utils
    import tensorflow.compat.v1 as tf
    from dslr_py import dslr
    from dslr_py.utils import MeasureDuration

    with MeasureDuration('Setup env'):
        config = tf.ConfigProto(device_count={'GPU': 1})
        res_sizes = utils.get_resolutions()
        init_obj = dslr.create_init_object(model_type, args.device)

    with MeasureDuration('Load model'):
        sess = tf.Session(config=config)
        sess = dslr.load_model(sess, model=model_type)

    source_dir = os.path.realpath(os.path.expanduser(args.source.rstrip('/'))) + '/'
    dest_dir = os.path.realpath(os.path.expanduser(args.destination.rstrip('/'))) + '/'
    config = utils.load_config(args.config)

    if config['s3']['upload']:
        log.info('S3 Uploading enabled')

    files = [f for f in os.listdir(source_dir) if os.path.isfile(source_dir + f) and f.rsplit(".", 1)[1] == 'jpg']

    if args.limit is not None:
        files = files[:args.limit]

    if len(files) < 1:
        log.info('No files to process in "%s". Exiting', source_dir)
        return

    uploader = utils.S3Helper(config)
    upload_pool = dslr.ManagedThreadPoolExecutor(4)

    files = list(filter(_f, files))
    log.info('%d files fround in "%s"', len(files), source_dir)
    directories_created = False

    for file in tqdm(files):
        file_name = file.rsplit(".", 1)[0]
        file_path = dest_dir + file_name + '.jpg'

        log.info('Generating image from original "%s"', source_dir + file)
        z = misc.imread(source_dir + file)
        original_size = z.shape
        image = np.float16(misc.imresize(z, res_sizes[model_type])) / 255
        image_crop = utils.extract_crop(image, "orig", model_type, res_sizes)
        image_crop_2d = np.reshape(image_crop, [1, init_obj['IMAGE_SIZE']])

        enhanced_2d = sess.run(init_obj['enhanced'], feed_dict={init_obj['x_']: image_crop_2d})
        enhanced_image = np.reshape(enhanced_2d, [init_obj['IMAGE_HEIGHT'], init_obj['IMAGE_WIDTH'], 3])

        with MeasureDuration('Save output'):
            if not directories_created:
                for img_info in config['sizes']:
                    if not os.path.exists(dest_dir + img_info['name']):
                        os.makedirs(dest_dir + img_info['name'])
                directories_created = True

            original_sized_enhanced = cv2.resize(enhanced_image, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)

            assert original_sized_enhanced.shape[0] == original_size[0]
            assert original_sized_enhanced.shape[1] == original_size[1]

            # Save Original size
            save_img_array(original_sized_enhanced * 255, file_path)

            for size_info in config['sizes']:
                if original_size[1] < size_info['width']:
                    sized_img = original_sized_enhanced
                    log.debug('Size %s is too small for %s', size_info['name'], original_size[1])
                else:
                    log.debug('Saving re-sized image: %s', file_path)
                    sized_img = utils.rescale_by_width(original_sized_enhanced, size_info['width'])
                file_path = dest_dir + size_info['name'] + '/' + args.version_name + '_' + file_name + '.jpg'

                try:
                    save_img_array(sized_img * 255, file_path)

                    if config['s3']['upload']:
                        upload_pool.submit(
                            utils.upload_completed_file,
                            uploader,
                            full_file_name=file,
                            size_name=size_info['name'],
                            version_name=args.version_name,
                            src_file_path=file_path
                        )
                except KeyboardInterrupt:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise

    log.info('Waiting for all uploads to be finished')

    while True:
        if upload_pool.done():
            break
        time.sleep(0.2)

    if not upload_pool.get_exception() is None:
        print('Some uploads has been failed', file=sys.stderr)
        for er in upload_pool.get_exceptions():
            logging.error("{}".format(er))
        exit(1)

    log.info('All done')
