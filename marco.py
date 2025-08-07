import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input
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
        pass

        
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


