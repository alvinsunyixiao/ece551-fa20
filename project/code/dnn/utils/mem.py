import tensorflow as tf

def set_memory_growth():
    devices = tf.config.experimental.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
