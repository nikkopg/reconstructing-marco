import tensorflow as tf


def load_checkpoint_variables(ckpt_path):
    ''' Load available weights in the savedmodel checkpoint '''
    reader = tf.train.load_checkpoint(ckpt_path)
    ckpt_vars = {name: reader.get_tensor(name) for name, _ in tf.train.list_variables(ckpt_path)}

    return ckpt_vars


def map_ckpt_to_keras(ckpt_name):
    ''' This is to map the weight's name in the savedmodel to match the name in Keras model '''

    # Stem layers
    if ckpt_name.startswith('Inception/Conv2d_') and ckpt_name.endswith('/weights'):
        return ckpt_name.replace('/weights', '/kernel:0')
    if ckpt_name.startswith('Inception/Conv2d_') and '/BatchNorm/' in ckpt_name:
        return ckpt_name + ':0'

    # Inception blocks and BatchNorm
    if ckpt_name.startswith('Inception/Mixed_') and ckpt_name.endswith('/weights'):
        return ckpt_name.replace('/weights', '/kernel:0')
    if ckpt_name.startswith('Inception/Mixed_') and '/BatchNorm/' in ckpt_name:
        return ckpt_name + ':0'
    
    if ckpt_name.startswith('Inception/Logits') and ckpt_name.endswith('/weights'):
        return ckpt_name.replace('/weights', '/kernel:0')
    if ckpt_name.startswith('Inception/Logits') and ckpt_name.endswith('/biases'):
        return ckpt_name.replace('/biases', '/bias:0')
    
    if ckpt_name.startswith('Inception/AuxLogits') and ckpt_name.endswith('/weights'):
        return ckpt_name.replace('/weights', '/kernel:0')
    if ckpt_name.startswith('Inception/AuxLogits') and ckpt_name.endswith('/biases'):
        return ckpt_name.replace('/biases', '/bias:0')
    if ckpt_name.startswith('Inception/AuxLogits/Conv2d_') and '/BatchNorm/' in ckpt_name:
        return ckpt_name + ':0'

    return None