import math
from ga_optimizer.utils import optimizers
from tensorflow import keras

from ga_optimizer.utils.utils import (
    is_tf_211_and_above,
    optimizer_has_legacy,
    os_is_mac,
)


def make_ga_optimizer(
    desired_batch_size,
    batch_size,
    base_optimizer,
    base_optimizer_params=None,
    log_level=optimizers.Optimizer.LOG_NONE,
):
    # Gradient accumulation steps are calculated to ensure that the effective batch size
    # matches or exceeds the desired batch size. This is particularly useful when the
    # hardware cannot handle the desired batch size in one go due to memory constraints.
    # By accumulating gradients over several smaller batches, we simulate the effect
    # of a larger batch size. The number of accumulation steps is the smallest number
    # of steps required to reach or exceed the desired batch size.

    # So accumulation_steps will be the number of steps it takes to reach one full simulated batch.

    # Calculate gradient accumulation steps dynamically
    # We use math.ceil to ensure we always round up to the nearest whole number

    accumulation_steps = math.ceil(desired_batch_size / batch_size)

    if os_is_mac() and not optimizer_has_legacy(base_optimizer):
        base_optimizer_name = base_optimizer.name
    else:
        base_optimizer_name = base_optimizer._name

    # Check if "legacy" is available in the optimizer namespace, if os is MacOS, if tf version is 211 and above and if the base optimizer is not already a legacy optimizer
    # If that's the case, we convert it to legacy now on the spot to avoid problems later
    if (
        optimizer_has_legacy()
        and os_is_mac()
        and is_tf_211_and_above()
        and not optimizer_has_legacy(base_optimizer)
    ):
        print(
            "You have passed a base optimizer that is not a legacy optimizer. Since you are using TensorFlow 2.11 or above and MacOs, it will be converted into a legacy optimizer."
        )
        print(
            "Make sure you pass the parameters (if you have any) to make_ga_optimizer()!"
        )
        legacy_optimizer_class = getattr(keras.optimizers.legacy, base_optimizer_name)
        if base_optimizer_params is None:
            base_optimizer_params = {}
        base_optimizer = legacy_optimizer_class(**base_optimizer_params)

    print("base_optimizer_name:", base_optimizer_name)
    print("Using optimizer wrapper for GA.")
    ga_optimizer = optimizers.Optimizer(
        name=base_optimizer_name,
        optimizer=base_optimizer,
        steps=accumulation_steps,
        log_level=log_level,
    )

    return ga_optimizer
