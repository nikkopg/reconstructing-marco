import numpy as np
import tensorflow as tf
from utils import map_ckpt_to_keras


def count_matched_weights(ckpt_vars, model):
    matched = 0
    for ckpt_name in ckpt_vars:    
        for var in model.model.variables:
            keras_name = map_ckpt_to_keras(ckpt_name)
            if keras_name == var.name and ckpt_vars[ckpt_name].shape == var.shape:
                # check if the weights value are the same
                if not tf.reduce_all(tf.equal(ckpt_vars[ckpt_name], var)).numpy():
                    print(f"Mismatch in values for {ckpt_name} ({ckpt_vars[ckpt_name].shape}) and {var.name} ({var.shape})")
                else:
                    # print(f"Values match for {ckpt_name} and {var.name}")
                    matched += 1

    return matched