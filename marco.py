from test import map_ckpt_to_keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Lambda, Input
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Concatenate


class MarcoWrapper():

    def __init__(self, depth_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)

        ''' 
        Original MARCO model has only 4 classes: Clear (C), Precipitate (P), Other (0), Crystal (X).
        The model is a variant of InceptionV3, but modified in few layers. As the paper discussed, 
        one of the modification was the maximum conv layer depths to reduce model complexity.
        '''
        self.num_classes = 4
        self.batch_size = 64
        self.num_training_samples = 442930  # total of training image mentioned in the paper, adjust as needed

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
        # self.output_layer = None
        self.model = self.reconstruct(depth_multiplier)
    

    def reconstruct(self, create_aux_logits=True):
        
        '''
        Full network architecture is shown in marco-arch.txt
        '''

        # Bigger input layer (599, 599, 3)
        inputs = Input(shape=(599, 599, 3), name="input")

        # Additional layer to compress larger input image, naming it as Conv2d_0a_3x3
        x = self.conv2d_bn(inputs, filters=16, kernel_size=(3, 3), strides=2, name="Inception/Conv2d_0a_3x3")

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

        '''
        InceptionV3 has 2 classication heads: Logits and Auxiliary Logits. Read papers for the details.
        The MARCO savedmodel weights also has these 2 classification heads.
        AuxLogits head only used for training. We reconstruct this head just to store the weights.
        '''
        # Auxiliary Classification Head (not used in this model to inference) -- branched from Mixed_6e
        if create_aux_logits:
            aux_logits = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid', name='Inception/Logits/AuxLogits/AvgPool_1a_5x5')(x)
            aux_logits = self.conv2d_bn(aux_logits, filters=128, kernel_size=(1, 1), name='Inception/AuxLogits/Conv2d_1b_1x1')
            aux_logits = self.conv2d_bn(aux_logits, filters=768, kernel_size=(5, 5), padding='valid', name='Inception/AuxLogits/Conv2d_2a_5x5')
            aux_logits = Conv2D(self.num_classes, (1, 1), name="Inception/AuxLogits/Conv2d_2b_1x1")(aux_logits)
            aux_logits = Lambda(lambda t: tf.squeeze(t, [1, 2]), name="Inception/AuxLogits/SpatialSqueeze")(aux_logits)
            aux_logits = Activation('softmax', name='Inception/AuxLogits/Predictions/Softmax')(aux_logits)

        # Mixed 7a, 7b, 7c blocks
        x = self.mixed_7a(x, self.max_depth["Mixed_7a"], name="Mixed_7a") 
        x = self.mixed_7b(x, self.max_depth["Mixed_7b"], name="Mixed_7b")
        x = self.mixed_7c(x, self.max_depth["Mixed_7c"], name="Mixed_7c")

        # Main Classification Head
        x = AveragePooling2D(pool_size=(8, 8), padding='valid', name="Inception/Logits/AvgPool_1a_8x8")(x)
        x = Conv2D(self.num_classes, (1, 1), name="Inception/Logits/Conv2d_1c_1x1")(x)
        x = Lambda(lambda t: tf.squeeze(t, [1, 2]), name="Inception/Logits/SpatialSqueeze")(x)
        x = Activation('softmax', name="Inception/Logits/Predictions/Softmax")(x)

        if create_aux_logits:
            model = Model(inputs, [x, aux_logits], name="MARCO_InceptionV3")
        else:
            model = Model(inputs, x, name='MARCO_InceptionV3')
        
        steps_per_epoch = self.num_training_samples // self.batch_size
        decay_epochs = 2
        decay_steps = steps_per_epoch * decay_epochs

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.045,
            decay_steps=decay_steps,
            decay_rate=0.94,
            staircase=True
        )

        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            rho=0.9,
            momentum=0.9,
            epsilon=0.1,
            centered=False  # default
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',  # adjust to your task, might need to recompile
            metrics=['accuracy']
        )

        return model
    

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
        return self.mixed_6c(x, max_depth, name=name)


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
    

    def mixed_7a(self, x, max_depth, name="Mixed_7a"):
        ''' Inception block as figure 7 in the paper (https://arxiv.org/pdf/1512.00567) '''

        d = lambda orig: min(int(orig * self.depth_multiplier), max_depth)

        # branch 0
        b0 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")
        b0 = self.conv2d_bn(b0, d(320), (3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_0/Conv2d_1a_3x3")

        # branch 1
        b1 = self.conv2d_bn(x, d(192), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1 = self.conv2d_bn(b1, d(192), (1, 7), name=f"Inception/{name}/Branch_1/Conv2d_0b_1x7")
        b1 = self.conv2d_bn(b1, d(192), (7, 1), name=f"Inception/{name}/Branch_1/Conv2d_0c_7x1")
        b1 = self.conv2d_bn(b1, d(192), (3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_1/Conv2d_1a_3x3")

        # branch 2 (MaxPool, no weights)
        b2 = MaxPooling2D((3, 3), strides=2, padding='valid', name=f"Inception/{name}/Branch_2/MaxPool_1a_3x3")(x)

        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2])


    def mixed_7b(self, x, max_depth, name="Mixed_7b"):
        d = lambda orig: min(int(orig * self.depth_multiplier), max_depth)

        # branch 0
        b0 = self.conv2d_bn(x, d(320), (1, 1), name=f"Inception/{name}/Branch_0/Conv2d_0a_1x1")

        # branch 1
        '''
        Mixed_7b and 7_b should've been exactly the same.
        but the savedmodel weights name for layer Branch_1/Conv2d_0x_3x1 are different
        Mixed_7b = 0b, while Mixed_7c = 0c. Typo?
        '''
        if name == "Mixed_7c":
            branch_layer_name = '0c'
        else:
            branch_layer_name = '0b'

        b1 = self.conv2d_bn(x, d(384), (1, 1), name=f"Inception/{name}/Branch_1/Conv2d_0a_1x1")
        b1a = self.conv2d_bn(b1, d(384), (1, 3), name=f"Inception/{name}/Branch_1/Conv2d_0b_1x3")
        b1b = self.conv2d_bn(b1, d(384), (3, 1), name=f"Inception/{name}/Branch_1/Conv2d_{branch_layer_name}_3x1")
        b1 = Concatenate(name=f"Inception/{name}/Branch_1/concat")([b1a, b1b])

        # branch 2
        b2 = self.conv2d_bn(x, d(448), (1, 1), name=f"Inception/{name}/Branch_2/Conv2d_0a_1x1")
        b2 = self.conv2d_bn(b2, d(384), (3, 3), name=f"Inception/{name}/Branch_2/Conv2d_0b_3x3")
        b2a = self.conv2d_bn(b2, d(384), (1, 3), name=f"Inception/{name}/Branch_2/Conv2d_0c_1x3")
        b2b = self.conv2d_bn(b2, d(384), (3, 1), name=f"Inception/{name}/Branch_2/Conv2d_0d_3x1")
        b2 = Concatenate(name=f"Inception/{name}/Branch_2/concat")([b2a, b2b])
        
        # branch 3
        b3 = AveragePooling2D((3, 3), strides=1, padding='same', name=f"Inception/{name}/Branch_3/AvgPool_0a_3x3")(x)
        b3 = self.conv2d_bn(b3, d(96), (1, 1), name=f"Inception/{name}/Branch_3/Conv2d_0b_1x1") # -> this layer is hacked

        return Concatenate(axis=3, name=f"Inception/{name}/concat")([b0, b1, b2, b3])


    def mixed_7c(self, x, max_depth, name="Mixed_7c"):
        return self.mixed_7b(x, max_depth, name=name)

        
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

        conv_name = f"{name}" if name else None
        bn_name = f"{name}/BatchNorm" if name else None
        relu_name = f"{name}/Relu" if name else None

        if channel_first:
            bn_axis = 1
        else:
            bn_axis = 3

        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)

        return x
    

    def load_weights(self, checkpoint_variables):
        assigned = 0
        for ckpt_name in checkpoint_variables:    
            for var in self.model.variables:
                keras_name = map_ckpt_to_keras(ckpt_name)
                if keras_name == var.name and checkpoint_variables[ckpt_name].shape == var.shape:
                    var.assign(checkpoint_variables[ckpt_name])
                    print(f"Assigned {ckpt_name} ({checkpoint_variables[ckpt_name].shape}) -> {var.name} ({var.shape})")
                    assigned += 1
                    break
            else:
                print(f"Skipped {ckpt_name} ({checkpoint_variables[ckpt_name].shape}) (no matching checkpoint variable found)")

        print(f"Total assigned: {assigned}/{len(checkpoint_variables)}")
