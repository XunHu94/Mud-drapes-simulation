import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import tfutil
# import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Parse individual image from a tfrecords file.

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_tf_float16(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.float16)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def parse_tfrecord_np_float16(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.float16).reshape(shape)

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        repeat          = True,     # Repeat dataset indefinitely.
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2,        # Number of concurrent threads.
        ):

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channel, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self._tf_minibatch_in   = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        # List realimage tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        #tfr_realimage_files = tfr_files[:-2] #as tfrecord files include 02-06 real image files, one prob_image file and one well_facies file, [:-2] ensures only reale image files are selected.
        tfr_realimage_files = tfr_files
        assert len(tfr_realimage_files) >= 1
        tfr_realimage_shapes = []  # tfr_realimage_shapes
        for tfr_realimage_file in tfr_realimage_files:  #
            tfr_realimage_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_realimage_file, tfr_realimage_opt):
                tfr_realimage_shapes.append(parse_tfrecord_np(record).shape)
                break


        # Determine shape and resolution of realimage. some parameters are marked with _realimage_, but some are not. All probimage related parameters are marked with _probimage_.
        max_realimage_shape = max(tfr_realimage_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_realimage_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_realimage_shape[0], self.resolution, self.resolution]
        tfr_realimage_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_realimage_shapes]
        assert all(shape[0] == max_realimage_shape[0] for shape in tfr_realimage_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_realimage_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_realimage_shapes, tfr_realimage_lods))
        assert all(lod in tfr_realimage_lods for lod in range(self.resolution_log2 - 1))

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])

            for tfr_realimage_file, tfr_realimage_shape, tfr_realimage_lod in zip(tfr_realimage_files, tfr_realimage_shapes, tfr_realimage_lods):
                if tfr_realimage_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_realimage_file, compression_type='', buffer_size=buffer_mb<<17)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.data.Dataset.zip((dset))
                bytes_per_item = np.prod(tfr_realimage_shape) * np.dtype(self.dtype).itemsize
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 17) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 17) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_realimage_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self, minibatch_size): # => images, probimages
        images = self._tf_iterator.get_next()
        return images

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, probimages
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf(minibatch_size)
        return tfutil.run(self._tf_minibatch_np)

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_image_tf(self, minibatch_size): # => images
        images = self._tf_iterator.get_next()
        return images

    # Get next minibatch as NumPy arrays.
    def get_minibatch_image_np(self, minibatch_size, lod=0): # => images
        self.configure(minibatch_size, lod)
        return tfutil.run(self.get_minibatch_image_tf(minibatch_size))

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name='dataset.TFRecordDataset', data_dir=None, verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
        adjusted_kwargs['tfrecord_dir'] = os.path.join(data_dir, adjusted_kwargs['tfrecord_dir'])
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = tfutil.import_obj(class_name)(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
    return dataset
