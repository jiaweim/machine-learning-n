# Module|tf.keras.utils

2022-03-04, 19:56
****

## 类

|类|说明|
|---|---|
|CustomObjectScope|Exposes custom classes/functions to Keras deserialization internals.|
|GeneratorEnqueuer|Builds a queue out of a data generator.|
|OrderedEnqueuer|Builds a Enqueuer from a Sequence.|
|Progbar|Displays a progress bar.|
|Sequence|Base object for fitting to a sequence of data, such as a dataset.|
|SequenceEnqueuer|Base class to enqueue inputs.|
|SidecarEvaluator|A class designed for a dedicated evaluator task.|
|custom_object_scope|Exposes custom classes/functions to Keras deserialization internals.|

## 函数

|函数|说明|
|---|---|
|array_to_img(...)|Converts a 3D Numpy array to a PIL Image instance.|
|deserialize_keras_object(...)|Turns the serialized form of a Keras object back into an actual object.|
|get_custom_objects(...)|Retrieves a live reference to the global dictionary of custom objects.|
|get_file(...)|Downloads a file from a URL if it not already in the cache.|
|get_registered_name(...)|Returns the name registered to an object within the Keras framework.|
|get_registered_object(...)|Returns the class associated with name if it is registered with Keras.|
|get_source_inputs(...)|Returns the list of input tensors necessary to compute tensor.|
|image_dataset_from_directory(...)|Generates a tf.data.Dataset from image files in a directory.|
|img_to_array(...)|Converts a PIL Image instance to a Numpy array.|
|load_img(...)|Loads an image into PIL format.|
|model_to_dot(...)|Convert a Keras model to dot format.|
|normalize(...)|Normalizes a Numpy array.|
|pack_x_y_sample_weight(...)|Packs user-provided data into a tuple.|
|plot_model(...)|Converts a Keras model to dot format and save to a file.|
|register_keras_serializable(...)|Registers an object with the Keras serialization framework.|
|save_img(...)|Saves an image stored as a Numpy array to a path or file object.|
|serialize_keras_object(...)|Serialize a Keras object into a JSON-compatible representation.|
|set_random_seed(...)|Sets all random seeds for the program (Python, NumPy, and TensorFlow).|
|[text_dataset_from_directory(...)](text_dataset_from_directory.md)|Generates a tf.data.Dataset from text files in a directory.|
|timeseries_dataset_from_array(...)|Creates a dataset of sliding windows over a timeseries provided as array.|
|to_categorical(...)|Converts a class vector (integers) to binary class matrix.|
|unpack_x_y_sample_weight(...)|Unpacks user-provided data tuple.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/utils
