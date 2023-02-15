# tf.keras.layers

2022-03-03, 23:10
****

## 类

|类|说明|
|---|---|
|AbstractRNNCell|Abstract object representing an RNN cell.|
|Activation|Applies an activation function to an output.|
|ActivityRegularization|Layer that applies an update to the cost function based input activity.|
|Add|Layer that adds a list of inputs.|
|AdditiveAttention|Additive attention layer, a.k.a. Bahdanau-style attention.|
|AlphaDropout|Applies Alpha Dropout to the input.|
|Attention|Dot-product attention layer, a.k.a. Luong-style attention.|
|Average|Layer that averages a list of inputs element-wise.|
|AveragePooling1D|Average pooling for temporal data.|
|AveragePooling2D|Average pooling operation for spatial data.|
|AveragePooling3D|Average pooling operation for 3D data (spatial or spatio-temporal).|
|AvgPool1D|Average pooling for temporal data.|
|AvgPool2D|Average pooling operation for spatial data.|
|AvgPool3D|Average pooling operation for 3D data (spatial or spatio-temporal).|
|BatchNormalization|Layer that normalizes its inputs.|
|[Bidirectional](Bidirectional.md)|双向 RNN 的 wrapper|
|CategoryEncoding|A preprocessing layer which encodes integer features.|
|CenterCrop|A preprocessing layer which crops images.|
|Concatenate|Layer that concatenates a list of inputs.|
|Conv1D|1D convolution layer (e.g. temporal convolution).|
|Conv1DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|Conv2D|2D convolution layer (e.g. spatial convolution over images).|
|Conv2DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|Conv3D|3D convolution layer (e.g. spatial convolution over volumes).|
|Conv3DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|ConvLSTM1D|1D Convolutional LSTM.|
|ConvLSTM2D|2D Convolutional LSTM.|
|ConvLSTM3D|3D Convolutional LSTM.|
|Convolution1D|1D convolution layer (e.g. temporal convolution).|
|Convolution1DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|Convolution2D|2D convolution layer (e.g. spatial convolution over images).|
|Convolution2DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|Convolution3D|3D convolution layer (e.g. spatial convolution over volumes).|
|Convolution3DTranspose|Transposed convolution layer (sometimes called Deconvolution).|
|Cropping1D|Cropping layer for 1D input (e.g. temporal sequence).|
|Cropping2D|Cropping layer for 2D input (e.g. picture).|
|Cropping3D|Cropping layer for 3D data (e.g. spatial or spatio-temporal).|
|Dense|Just your regular densely-connected NN layer.|
|DenseFeatures|A layer that produces a dense Tensor based on given feature_columns.|
|DepthwiseConv1D|Depthwise 1D convolution.|
|DepthwiseConv2D|Depthwise 2D convolution.|
|Discretization|A preprocessing layer which buckets continuous features by ranges.|
|Dot|Layer that computes a dot product between samples in two tensors.|
|Dropout|Applies Dropout to the input.|
|ELU|Exponential Linear Unit.|
|Embedding|Turns positive integers (indexes) into dense vectors of fixed size.|
|[Flatten](Flatten.md)|Flattens the input. Does not affect the batch size.|
|GRU|Gated Recurrent Unit - Cho et al. 2014.|
|GRUCell|Cell |for the GRU layer.|
|GaussianDropout|Apply multiplicative 1-centered Gaussian noise.|
|GaussianNoise|Apply additive zero-centered Gaussian noise.|
|GlobalAveragePooling1D|Global average pooling operation for temporal data.|
|GlobalAveragePooling2D|Global average pooling operation for spatial data.|
|GlobalAveragePooling3D|Global Average pooling operation for 3D data.|
|GlobalAvgPool1D|Global average pooling operation for temporal data.|
|GlobalAvgPool2D|Global average pooling operation for spatial data.|
|GlobalAvgPool3D|Global Average pooling operation for 3D data.|
|GlobalMaxPool1D|Global max pooling operation for 1D temporal data.|
|GlobalMaxPool2D|Global max pooling operation for spatial data.|
|GlobalMaxPool3D|Global Max pooling operation for 3D data.|
|GlobalMaxPooling1D|Global max pooling operation for 1D temporal data.|
|GlobalMaxPooling2D|Global max pooling operation for spatial data.|
|GlobalMaxPooling3D|Global Max pooling operation for 3D data.|
|Hashing|A preprocessing layer which hashes and bins categorical features.|
|InputLayer|Layer to be used as an entry point into a Network (a graph of layers).|
|InputSpec|Specifies the rank, dtype and shape of every input to a layer.|
|IntegerLookup|A preprocessing layer which maps integer features to contiguous ranges.|
|LSTM|Long Short-Term Memory layer - Hochreiter 1997.|
|LSTMCell|Cell |for the LSTM layer.|
|Lambda|Wraps arbitrary expressions as a Layer object.|
|Layer|This is the |from which all layers inherit.|
|LayerNormalization|Layer normalization layer (Ba et al., 2016).|
|LeakyReLU|Leaky version of a Rectified Linear Unit.|
|LocallyConnected1D|Locally-connected layer for 1D inputs.|
|LocallyConnected2D|Locally-connected layer for 2D inputs.|
|[Masking](Masking.md)|Masks a sequence by using a mask value to skip timesteps.|
|MaxPool1D|Max pooling operation for 1D temporal data.|
|MaxPool2D|Max pooling operation for 2D spatial data.|
|MaxPool3D|Max pooling operation for 3D data (spatial or spatio-temporal).|
|MaxPooling1D|Max pooling operation for 1D temporal data.|
|MaxPooling2D|Max pooling operation for 2D spatial data.|
|MaxPooling3D|Max pooling operation for 3D data (spatial or spatio-temporal).|
|Maximum|Layer that computes the maximum (element-wise) a list of inputs.|
|Minimum|Layer that computes the minimum (element-wise) a list of inputs.|
|MultiHeadAttention|MultiHeadAttention layer.|
|Multiply|Layer that multiplies (element-wise) a list of inputs.|
|Normalization|A preprocessing layer which normalizes continuous features.|
|PReLU|Parametric Rectified Linear Unit.|
|Permute|Permutes the dimensions of the input according to a given pattern.|
|RNN|Base |for recurrent layers.|
|RandomContrast|A preprocessing layer which randomly adjusts contrast during training.|
|RandomCrop|A preprocessing layer which randomly crops images during training.|
|RandomFlip|A preprocessing layer which randomly flips images during training.|
|RandomHeight|A preprocessing layer which randomly varies image height during training.|
|RandomRotation|A preprocessing layer which randomly rotates images during training.|
|RandomTranslation|A preprocessing layer which randomly translates images during training.|
|RandomWidth|A preprocessing layer which randomly varies image width during training.|
|RandomZoom|A preprocessing layer which randomly zooms images during training.|
|ReLU|Rectified Linear Unit activation function.|
|RepeatVector|Repeats the input n times.|
|Rescaling|A preprocessing layer which rescales input values to a new range.|
|Reshape|Layer that reshapes inputs into the given shape.|
|Resizing|A preprocessing layer which resizes images.|
|SeparableConv1D|Depthwise separable 1D convolution.|
|SeparableConv2D|Depthwise separable 2D convolution.|
|SeparableConvolution1D|Depthwise separable 1D convolution.|
|SeparableConvolution2D|Depthwise separable 2D convolution.|
|SimpleRNN|Fully-connected RNN where the output is to be fed back to input.|
|SimpleRNNCell|Cell |for SimpleRNN.|
|Softmax|Softmax activation function.|
|SpatialDropout1D|Spatial 1D version of Dropout.|
|SpatialDropout2D|Spatial 2D version of Dropout.|
|SpatialDropout3D|Spatial 3D version of Dropout.|
|StackedRNNCells|Wrapper allowing a stack of RNN cells to behave as a single cell.|
|StringLookup|A preprocessing layer which maps string features to integer indices.|
|Subtract|Layer that subtracts two inputs.|
|TextVectorization|A preprocessing layer which maps text features to integer sequences.|
|ThresholdedReLU|Thresholded Rectified Linear Unit.|
|TimeDistributed|This wrapper allows to apply a layer to every temporal slice of an input.|
|UpSampling1D|Upsampling layer for 1D inputs.|
|UpSampling2D|Upsampling layer for 2D inputs.|
|UpSampling3D|Upsampling layer for 3D inputs.|
|Wrapper|Abstract wrapper base class.|
|ZeroPadding1D|Zero-padding layer for 1D input (e.g. temporal sequence).|
|ZeroPadding2D|Zero-padding layer for 2D input (e.g. picture).|
|ZeroPadding3D|Zero-padding layer for 3D data (spatial or spatio-temporal).|

## 函数

|函数|说明|
|---|---|
|Input(...)|Input() is used to instantiate a Keras tensor.|
|add(...)|Functional interface to the tf.keras.layers.Add layer.|
|average(...)|Functional interface to the tf.keras.layers.Average layer.|
|concatenate(...)|Functional interface to the Concatenate layer.|
|deserialize(...)|Instantiates a layer from a config dictionary.|
|dot(...)|Functional interface to the Dot layer.|
|maximum(...)|Functional interface to compute maximum (element-wise) list of inputs.|
|minimum(...)|Functional interface to the Minimum layer.|
|multiply(...)|Functional interface to the Multiply layer.|
|serialize(...)|Serializes a Layer object into a JSON-compatible representation.|
|subtract(...)|Functional interface to the Subtract layer.|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers
