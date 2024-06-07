import tensorflow as tf 

def get_scalar_run_tensorboard(tag, filepath):
    values,steps = [],[]
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value) > 0:
            if e.summary.value[0].tag == tag:
                value, step = (e.summary.value[0].simple_value, e.step)
                values.append(value)
                steps.append(step)
    return values, steps