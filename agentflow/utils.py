import tensorflow as tf

def check_whats_connected(output):
    for v in tf.global_variables():
        g = tf.gradients([output],v)[0]
        if g is None:
            print('NONE    ',v.name)
        else:
            print('GRADIENT',v.name)
