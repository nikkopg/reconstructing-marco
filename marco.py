import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Concatenate


class Marco(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ''' 
        Original MARCO model has only 4 classes: Clear (C), Precipitate (P), Other (0), Crystal (X).
        The model is a variant of InceptionV3, but modified in few layers. As the paper discussed, 
        one of the modification was the maximum conv layer depths to reduce model complexity.
        '''
        self.num_classes = 4
        self.max_depth = {
            "Conv2d_4a_3x3": 144,
            "Mixed_6b": 128,
            "Mixed_6c": 144,
            "Mixed_6d": 144,
            "Mixed_6e": 96,
            "Mixed_7a": 96,
            "Mixed_7b": 192,
            "Mixed_7c": 192,
        }
        

        # build model
        self.input_layer = None
        self.output_layer = None

    
    def call(self, inputs):
        
        # x = self.model(inputs)

        return self.output_layer # (x)
    

    def build(self, depth_multiplier=1.0, create_aux_logits=True):
        
        '''
        Full network architecture is shown in model-arch.txt
        '''

        # Bigger input layer (599, 599, 3)
        self.input_layer = Input(shape=(599, 599, 3), name="input")

        # Additional layer to compress larger input image, naming it as Conv2d_0a_3x3
        x = self.conv2d_bn(self.input_layer, filters=16, kernel_size=(3, 3), strides=2, name="Inception/Conv2d_0a_3x3")

        # First convolutional layer block
        x = self.conv2d_bn(x, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', name="Inception/Conv2d_1a_3x3")
        x = self.conv2d_bn(x, filters=32, kernel_size=(3, 3), padding='valid', name="Inception/Conv2d_2a_3x3")
        x = self.conv2d_bn(x, filters=64, kernel_size=(3, 3), strides=1, name="Inception/Conv2d_2b_3x3")
        x = MaxPooling2D((3, 3), strides=2, padding='valid', name="Inception/Inception/MaxPool_3a_3x3/MaxPool")(x)

        # Second convolutional layer block
        x = self.conv2d_bn(x, filters=80, kernel_size=(1, 1), padding='valid', name='Inception/Conv2d_3b_1x1')
        x = self.conv2d_bn(x, filters=self.max_depth['Conv2d_4a_3x3'], kernel_size=(3, 3), padding='valid', name='Inception/Conv2d_4a_3x3')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name="Inception/Inception/MaxPool_5a_3x3/MaxPool")(x)

        # Mixed 5b, 5c, 5d blocks
        x = self.mixed_5_block(x, depth_multiplier, name="Mixed_5b")
        x = self.mixed_5_block(x, depth_multiplier, name="Mixed_5c")
        x = self.mixed_5_block(x, depth_multiplier, name="Mixed_5d")

        return Model(self.input_layer, x, name='MARCO_InceptionV3')
    

    def mixed_5_block(self, x, depth_multiplier, name="Mixed_5"):
        ''' Inception block 1 as figure 4 in the paper '''
        d = lambda orig: int(orig * depth_multiplier)

        # branch 0
        b0 = self.conv2d_bn(x, d(64), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")

        # branch 1
        if name == "Mixed_5c":
            sublayer_names = ["Conv2d_0b", "Conv_1_0c"]
        else:
            sublayer_names = ["Conv2d_0a", "Conv2d_0b"]

        b1 = self.conv2d_bn(x, d(48), (1, 1), name=f"Inception/{name}/Branch_1/{sublayer_names[0]}_1x1")
        b1 = self.conv2d_bn(b1, d(64), (5, 5), name=f"Inception/{name}/Branch_1/{sublayer_names[1]}_5x5")

        # branch 2
        b2 = self.conv2d_bn(x, d(64), (1, 1), name=f"Inception/{name}/Branch_2/Conv2d_0a_1x1")
        b2 = self.conv2d_bn(b2, d(96), (3, 3), name=f"Inception/{name}/Branch_2/Conv2d_0b_3x3")
        b2 = self.conv2d_bn(b2, d(96), (3, 3), name=f"Inception/{name}/Branch_2/Conv2d_0c_3x3")

        # branch 3
        if name == "Mixed_5b":
            layer_depth = d(32)
        else:
            layer_depth = d(64)

        b3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f"Inception/{name}/Branch_3/AvgPool_0a_3x3")(x)
        b3 = self.conv2d_bn(b3, layer_depth, (1, 1), name=f"Inception/{name}/Branch_3/Conv2d_0b_1x1")

        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2, b3])

        
    def conv2d_bn(self, x, filters, kernel_size, strides=1, padding='same', channel_first=False, name=None):
        '''
        This the similar function as conv2d_bn in the:
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/inception_v3.py#L381

        This function applies a 2d conv followed by batch norm and relu activation.
        Args:
            x           : input tensor
            filters     : number of filters for the conv2d
            kernel_size : size of the convolution kernel
            strides     : strides for the convolution
            padding     : padding type for the convolution
            name        : name for the layer. each layer will be appended with 
                          `/Conv2d`, `/BatchNorm`, and `/Relu` to match the weights name
                          in the original MARCO savedmodel.
        Returns:
            x           : output tensor after conv2d, batch norm and relu activation
        '''

        conv_name = f"{name}/Conv2d" if name else None
        bn_name = f"{name}/BatchNorm" if name else None
        relu_name = f"{name}/Relu" if name else None

        if channel_first:
            bn_axis = 1
        else:
            bn_axis = 3

        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)

        return x


