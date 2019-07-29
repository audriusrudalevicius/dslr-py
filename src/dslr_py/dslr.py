import functools
import logging
import os
from concurrent import futures

import numpy as np
import tensorflow.compat.v1 as tf
from scipy import misc
from tqdm import tqdm

from dslr_py import utils
from dslr_py.utils import MeasureDuration
from .models import resnet

resolution = "orig"


class Task:
    def __init__(self, threads_count=4):
        self.threads_count = threads_count
        self.pool = ManagedThreadPoolExecutor(self.threads_count)
        self.logger = logging.getLogger(__name__)
        self.task_fn = None
        self.on_item_done_callback = None

    def startup(self):
        pass

    def done(self):
        return self.pool.done()

    def submit(self, items, items_count, res_sizes=None):
        return self.pool.map(
            functools.partial(
                self.task_fn,
                res_sizes=res_sizes,
                callback=self.on_item_done_callback,
                logger=self.logger
            ),
            items
        )

    def shutdown(self):
        self.pool.shutdown()


class PrepareTask(Task):
    def __init__(self, threads_count, device, model):
        Task.__init__(self, threads_count)
        self.config = tf.ConfigProto(device_count={'GPU': 1})
        self.config.gpu_options.allow_growth = False

        self.task_fn = PrepareTask.process
        self.res_sizes = utils.get_resolutions()
        self.init_obj = None
        self.sess = None
        self.device = device
        self.model = model

    def startup(self):
        super().startup()
        self.init_obj = create_init_object(self.model, tf_device=self.device)
        self.sess = load_model(tf.Session(config=None), model=self.model)

    @staticmethod
    def process(item, sess=None, init_obj=None, res_sizes=None, logger=None, model=None, callback=None):

        with MeasureDuration('Prepare'):
            source_dir, file, dest_dir = item

            image = np.float16(misc.imresize(misc.imread(source_dir + file), res_sizes[model])) / 255
            image_crop = utils.extract_crop(image, "orig", model, res_sizes)
            image_crop_2d = np.reshape(image_crop, [1, init_obj['IMAGE_SIZE']])
            enhanced_2d = sess.run(init_obj['enhanced'], feed_dict={init_obj['x_']: image_crop_2d})
            enhanced_image = np.reshape(enhanced_2d, [init_obj['IMAGE_HEIGHT'], init_obj['IMAGE_WIDTH'], 3])
        if callback is not None:
            callback()

        return {
            'result': enhanced_image, 'file': file, 'dest_dir': dest_dir,
            'IMAGE_HEIGHT': init_obj['IMAGE_HEIGHT'],
            'IMAGE_WIDTH': init_obj['IMAGE_WIDTH']
        }

    def submit(self, items, items_count, res_sizes=None):
        return self.pool.map(
            functools.partial(
                self.task_fn,
                model=self.model,
                res_sizes=self.res_sizes,
                sess=self.sess,
                init_obj=self.init_obj,
                callback=self.on_item_done_callback,
                logger=self.logger
            ),
            items
        )


class SaveTask(Task):
    def __init__(self, threads_count, sizes):
        Task.__init__(self, threads_count)
        self.task_fn = SaveTask.process
        self.sizes = sizes

    @staticmethod
    def process(item, sizes, logger=None, callback=None):
        enhanced_image = item['result']
        resized_imgs = []

        with MeasureDuration('Resize'):
            for size in sizes:
                if enhanced_image.shape[1] < size['width']:
                    continue
                sized_img = utils.rescale_by_width(enhanced_image, size['width'])
                resized_imgs.append({'name': size['name'], 'img': sized_img})

        file_names = []
        with MeasureDuration('Save'):
            file_name = item['file'].rsplit(".", 1)[0]

            for img_info in resized_imgs:
                if not os.path.exists(item['dest_dir'] + img_info['name']):
                    os.makedirs(item['dest_dir'] + img_info['name'])

            for img_info in resized_imgs:
                full_file_path = item['dest_dir'] + img_info['name'] + "/v1_" + file_name + '.jpg'
                file_names.append({'size': img_info['name'], 'full_path': full_file_path, 'file_name': item['file']})
                try:
                    misc.imsave(full_file_path, img_info['img'])
                except KeyboardInterrupt:
                    if os.path.exists(full_file_path):
                        os.remove(full_file_path)
                    raise
        if callback is not None:
            callback()

        return item['file'], file_names

    def submit(self, items, items_count, res_sizes=None):
        return self.pool.map(
            functools.partial(
                self.task_fn,
                sizes=self.sizes,
                callback=self.on_item_done_callback,
                logger=self.logger
            ),
            items
        )


class CanceledException(Exception):
    pass


class ManagedThreadPoolExecutor(futures.ThreadPoolExecutor):
    def __init__(self, max_workers):
        futures.ThreadPoolExecutor.__init__(self, max_workers)
        self._futures = []

    def submit(self, fn, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        self._futures.append(future)
        return future

    def done(self):
        return all([x.done() for x in self._futures])

    def get_exceptions(self):
        l = []
        for x in self._futures:
            if x.exception():
                l.append(x.exception())
        return l

    def get_exception(self):
        for x in self._futures:
            if x.exception():
                return x.exception()
        return None


class Processor:
    def __init__(self, items, items_count, config, seed=42, device="/gpu:0"):
        self.logger = logging.getLogger('processor')
        self.items = items
        self.items_count = items_count
        self.seed = seed
        self.config = config
        self.tasks = [
            PrepareTask(1, self.device, self.config['model']),
            SaveTask(2, self.config['sizes'])
        ]
        self.device = device

    def start(self):
        res_sizes = utils.get_resolutions()
        t = tqdm(total=self.items_count)

        items = self.items
        self.tasks[-1].on_item_done_callback = t.update
        for task in self.tasks:
            task.startup()
            items = task.submit(items,
                                self.items_count,
                                res_sizes=res_sizes)

        for it in items:
            yield it


def create_init_object(model="iphone_orig", tf_device='/gpu:0'):
    if tf_device is not None:
        tf.device(tf_device)
    res_sizes = utils.get_resolutions()
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, model, resolution)

    # create placeholders for input images
    x_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # generate enhanced image
    enhanced = resnet(x_image)

    return {
        'enhanced': enhanced,
        'x_': x_,
        'x_image': x_image,
        'IMAGE_HEIGHT': IMAGE_HEIGHT,
        'IMAGE_WIDTH': IMAGE_WIDTH,
        'IMAGE_SIZE': IMAGE_SIZE
    }


def load_model(sess, model="iphone_orig"):
    saver = tf.train.Saver()
    saver.restore(sess, utils.resolve_pkg_data_path('models_orig/' + model))

    return sess
