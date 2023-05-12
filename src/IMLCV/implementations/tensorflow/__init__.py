try:
    import tensorflow as tf

    # tf.keras.backend.set_floatx("float64")
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"

    from jax.interpreters import batching
    from jax.experimental.jax2tf.call_tf import call_tf_p
    import functools
    from IMLCV.external.tf2jax import loop_batcher

    batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)


except (ImportError, ModuleNotFoundError):
    # Invalid device or cannot modify virtual devices once initialized.
    pass
