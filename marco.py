import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Concatenate


class Marco(tf.keras.Model):

    def __init__(self, depth_multiplier, **kwargs):
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
        self.depth_multiplier = depth_multiplier
        self.input_layer = None
        self.output_layer = None
        self.model = self.build(depth_multiplier)

    
    def call(self, inputs):
        '''  '''
        return self.model(inputs)
    

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
        x = self.mixed_5_block(x, name="Mixed_5b")
        x = self.mixed_5_block(x, name="Mixed_5c")
        x = self.mixed_5_block(x, name="Mixed_5d")

        # Mixed 6a, 6b, 6c, 6d, 6e blocks
        x = self.mixed_6a(x, name="Mixed_6a")
        x = self.mixed_6b(x, self.max_depth["Mixed_6b"], name="Mixed_6b")
        x = self.mixed_6c(x, self.max_depth["Mixed_6c"], name="Mixed_6c")
        x = self.mixed_6d(x, self.max_depth["Mixed_6d"], name="Mixed_6d")
        x = self.mixed_6e(x, self.max_depth["Mixed_6e"], name="Mixed_6e")

        return Model(self.input_layer, x, name='MARCO_InceptionV3')
    

    def mixed_5_block(self, x, name="Mixed_5"):
        ''' Inception block 1 as figure 4 in the paper '''
        d = lambda orig: int(orig * self.depth_multiplier)

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
    

    def mixed_6a(self, x, name="Mixed_6a"):
        d = lambda orig: int(orig * self.depth_multiplier)

        # branch 0
        b0 = self.conv2d_bn(x, d(256), (3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_0/Conv2d_1a_1x1")   # this layer is hacked

        # branch 1
        b1 = self.conv2d_bn(x, d(64), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1 = self.conv2d_bn(b1, d(96), (3, 3), name=f"Inception/{name}/Branch_1/Conv2d_0b_3x3")
        b1 = self.conv2d_bn(b1, d(96), (3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_1/Conv2d_1a_1x1")

        # branch 2 (MaxPool, no weights)
        b2 = MaxPooling2D((3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_2/MaxPool_1a_3x3")(x)

        # Concatenate all branches
        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2])


    def mixed_6b(self, x, max_depth, name="Mixed_6b"):
        ''' Inception block as figure 6 in the paper '''

        d = lambda orig: min(int(orig * self.depth_multiplier), max_depth)

        # branch 0
        b0 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")
        
        # branch 1
        b1 = self.conv2d_bn(x, d(128), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1 = self.conv2d_bn(b1, d(128), (1, 7), name=f"Inception/{name}/Branch_1/Conv2d_0b_1x7")
        b1 = self.conv2d_bn(b1, d(192), (7, 1), name=f"Inception/{name}/Branch_1/Conv2d_0c_7x1")

        # branch 2
        b2 = self.conv2d_bn(x, d(128), (1, 1), name=f"Inception/{name}/Branch_2/Conv2d_0a_1x1")
        b2 = self.conv2d_bn(b2, d(128), (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0b_7x1")
        b2 = self.conv2d_bn(b2, d(128), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0c_1x7")
        b2 = self.conv2d_bn(b2, d(128), (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0d_7x1")
        b2 = self.conv2d_bn(b2, d(192), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0e_1x7")

        # branch 3
        b3 = AveragePooling2D((3, 3), strides=1, padding='same', name=f"Inception/{name}/Branch_3/AvgPool_0a_3x3")(x)
        b3 = self.conv2d_bn(b3, d(192), (1, 1), name=f"Inception/{name}/Branch_3/Conv2d_0b_1x1")

        # Concatenate all branches
        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2, b3])


    def mixed_6c(self, x, max_depth, name="Mixed_6c"):
        ''' Inception block as figure 6 in the paper '''
        d = lambda orig: min(int(orig * self.depth_multiplier), max_depth)

        # branch 0
        b0 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")
        
        # branch 1
        b1 = self.conv2d_bn(x, d(160), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1 = self.conv2d_bn(b1, d(160), (1, 7), name=f"Inception/{name}/Branch_1/Conv2d_0b_1x7")
        b1 = self.conv2d_bn(b1, d(192), (7, 1), name=f"Inception/{name}/Branch_1/Conv2d_0c_7x1")

        # branch 2
        if name == "Mixed_6d":
            depth = d(160)
        else:
            depth = 160
            
        b2 = self.conv2d_bn(x, d(160), (1, 1), name=f"Inception/{name}/Branch_2/Conv2d_0a_1x1")
        b2 = self.conv2d_bn(b2, depth, (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0b_7x1") # this layer is hacked
        b2 = self.conv2d_bn(b2, d(160), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0c_1x7")
        b2 = self.conv2d_bn(b2, d(160), (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0d_7x1")
        b2 = self.conv2d_bn(b2, d(192), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0e_1x7")

        # branch 3
        b3 = AveragePooling2D((3, 3), strides=1, padding='same', name=f"Inception/{name}/Branch_3/AvgPool_0a_3x3")(x)
        b3 = self.conv2d_bn(b3, d(192), (1, 1), name=f"Inception/{name}/Branch_3/Conv2d_0b_1x1")

        # Concatenate all branches
        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2, b3])


    def mixed_6d(self, x, max_depth, name="Mixed_6d"):
        ''' Inception block as figure 6 in the paper. Similar to 6c'''
        return self.inception_6c(x, max_depth, name=name)


    def mixed_6e(self, x, max_depth, name="Mixed_6e"):
        ''' Inception block as figure 6 in the paper '''
        d = lambda orig: min(int(orig * self.depth_multiplier), max_depth)

        # branch 0
        b0 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")
        
        # branch 1
        b1 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1 = self.conv2d_bn(b1, d(192), (1, 7), name=f"Inception/{name}/Branch_1/Conv2d_0b_1x7")
        b1 = self.conv2d_bn(b1, d(192), (7, 1), name=f"Inception/{name}/Branch_1/Conv2d_0c_7x1")

        # branch 2
        b2 = self.conv2d_bn(x, 192, (1, 1), name=f"Inception/{name}/Branch_2/Conv2d_0a_1x1")      # this layer is hacked
        b2 = self.conv2d_bn(b2, d(192), (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0b_7x1")
        b2 = self.conv2d_bn(b2, d(192), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0c_1x7")
        b2 = self.conv2d_bn(b2, d(192), (7, 1), name=f"Inception/{name}/Branch_2/Conv2d_0d_7x1")
        b2 = self.conv2d_bn(b2, d(192), (1, 7), name=f"Inception/{name}/Branch_2/Conv2d_0e_1x7")

        # branch 3
        b3 = AveragePooling2D((3, 3), strides=1, padding='same', name=f"Inception/{name}/Branch_3/AvgPool_0a_3x3")(x)
        b3 = self.conv2d_bn(b3, d(192), (1, 1), name=f"Inception/{name}/Branch_3/Conv2d_0b_1x1")

        # Concatenate all branches
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


