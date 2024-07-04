import platform
import tensorflow as tf


def is_tf_211_and_above():
    """
    Check if the TensorFlow version is 2.11 or above.

    Returns:
        bool: True if TensorFlow version is 2.11 or above, False otherwise.
    """
    # Get Tensorflow version
    tf_version = tf.__version__.split(".")
    tf_major_version = int(tf_version[0])
    tf_minor_version = int(tf_version[1])

    # Check if TF version is 2.11 or above
    if tf_major_version == 2 and tf_minor_version >= 11:
        return True
    else:
        return False


def optimizer_has_legacy(optimizer=None):
    """
    Check if the specified optimizer or any optimizer has legacy available.

    Args:
        optimizer (tf.keras.optimizers.Optimizer, optional): Specific optimizer to check for legacy support. Defaults to None.

    Returns:
        bool: True if the specified optimizer or any optimizer has legacy available, False otherwise.
    """

    # Check if a a specific optimizer is legacy or not
    if optimizer:
        return _check_specific_optimizer_for_legacy(optimizer)

    # Check if the legacy optimizer is available
    return _check_optimizer_for_legacy()


def _check_optimizer_for_legacy():
    """
    Check if the legacy optimizer is available in TensorFlow.

    Returns:
        bool: True if legacy optimizer is available, False otherwise.
    """
    if hasattr(tf.keras.optimizers, "legacy"):
        return True
    else:
        return False


def _check_specific_optimizer_for_legacy(optimizer):
    """
    Check if a specific optimizer is a legacy optimizer.

    Args:
        optimizer (tf.keras.optimizers.Optimizer): Specific optimizer to check.

    Returns:
        bool: True if the optimizer is a legacy optimizer, False otherwise.
    """
    if "legacy" in str(optimizer.__class__):
        return True
    else:
        return False


def should_use_legacy_optimizer():
    """
    Determine if a legacy optimizer should be used based on TensorFlow version and optimizer availability.

    Returns:
        bool: True if TensorFlow version is 2.11 or above and a legacy optimizer is available, False otherwise.
    """
    return is_tf_211_and_above() and optimizer_has_legacy()


def os_is_windows():
    """
    Check if the operating system is Windows.

    Returns:
        bool: True if the operating system is Windows, False otherwise.
    """
    return platform.system() == "Windows"


def os_is_mac():
    """
    Check if the operating system is MacOS.

    Returns:
        bool: True if the operating system is MacOS, False otherwise.
    """
    return platform.system() == "Darwin"


def is_arm_mac():
    return platform.system() == "Darwin" and platform.processor() == "arm"


def os_is_linux():
    """
    Check if the operating system is Linux.

    Returns:
        bool: True if the operating system is Linux, False otherwise.
    """
    return platform.system() == "Linux"
