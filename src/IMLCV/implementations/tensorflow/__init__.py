try:
    import logging

    import tensorflow as tf

    tf.get_logger().setLevel(logging.WARN)

    logging.getLogger("tensorflow").addFilter(
        logging.Filter(
            "Compiled the loaded model, but the compiled metrics have yet to be built.",
        ),
    )
    logging.getLogger("tensorflow").addFilter(
        logging.Filter(
            "No training configuration found in the save file, so the model was *not* compiled. Compile it manually.",
        ),
    )

    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"

except (ImportError, ModuleNotFoundError):
    pass
