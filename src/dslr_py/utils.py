import hashlib
import json
import logging
import os
import re
import time
from functools import reduce

import boto3
import cv2
import numpy as np
import pkg_resources
import scipy.stats as st
import tensorflow.compat.v1 as tf

timer_log = logging.getLogger('timer')
upload_log = logging.getLogger('uploader')


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')


def get_resolutions():
    # IMAGE_HEIGHT, IMAGE_WIDTH
    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["iphone_orig"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["blackberry_orig"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["sony_orig"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes


def get_specified_res(res_sizes, phone, resolution):
    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


def extract_crop(image, resolution, phone, res_sizes):
    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up: y_down, x_up: x_down, :]


def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)


class S3Helper:
    def __init__(self, config):
        self.bucket_name = config['s3']['bucket_name']
        self.path_format = config['s3']['path_format']
        self.extra_args = config['s3']['extra_args']
        self.name_hash_format = config['s3']['name_hash_format']
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv(
                'APP_S3_ACCESS_KEY',
                config['s3']['ACCESS_KEY']
            ),
            aws_secret_access_key=os.getenv(
                'APP_S3_SECRET_KEY',
                config['s3']['SECRET_KEY']
            )
        )

    def upload(self, source_path, destination):
        return self.s3.upload_file(source_path, self.bucket_name, destination, self.extra_args)

    def format_path(self, file, size_name, version):
        file_name = file.rsplit(".", 1)[0]
        file_ext = file.rsplit(".", 1)[1]

        new_name = re.sub(
            r'^(\w{3})(\w{3})(\w{26})$', r'\1/\2/\3',
            hashlib.md5(self.name_hash_format.format(file_name, version).encode('utf-8')).hexdigest()
        )

        return self.path_format.format(size_name, version, new_name + '.' + file_ext)


class MeasureDuration:
    def __init__(self, title):
        self.start = None
        self.end = None
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        timer_log.debug("Total time taken for %s: %s", self.title, self.duration())

    def duration(self):
        return str((self.end - self.start) * 1000) + ' milliseconds'


def resolve_pkg_data_path(sub_path):
    return pkg_resources.resource_filename('dslr_py', 'data/' + sub_path)


def load_config(config_path=None):
    if config_path is None:
        config_path = resolve_pkg_data_path('default_config.json')
    with open(config_path) as f:
        return json.load(f)


def upload_completed_file(uploader, full_file_name=None, size_name=None, version_name=None, src_file_path=None):
    try:
        upload_path = uploader.format_path(full_file_name, size_name, version_name)
        upload_log.debug('Uploading resized image "%s" to "%s"', src_file_path, upload_path)
        uploader.upload(src_file_path, upload_path)
        upload_log.info('Uploaded image "%s" to "%s"', src_file_path, upload_path)

        upload_path2 = uploader.format_path(full_file_name, size_name, 'v1')
        upload_log.debug('Uploading alternative resized image "%s" to "%s"', src_file_path, upload_path2)
        uploader.upload(src_file_path, upload_path2)
        upload_log.info('Uploaded alternative image "%s" to "%s"', src_file_path, upload_path2)
    except Exception as e:
        upload_log.error('Failed to upload file to s3: %s', src_file_path)
        if os.path.exists(src_file_path):
            os.remove(src_file_path)
        raise
