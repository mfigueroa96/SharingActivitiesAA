import tensorflow as tf

taylorGraph = tf.Graph()
with taylorGraph.as_default():
    x = tf.constant(0.0174533, dtype=tf.float64) # EN RADIANES
    n = tf.Variable(-2.0, dtype=tf.float64)
    approx = tf.Variable(0.0, dtype=tf.float64)
    sign = tf.Variable(1.0, dtype=tf.float64)
    next_n = tf.assign(n, n + 2.0)
    next_sign = tf.assign(sign, (pow(-1, tf.cast(n, tf.float64) / 2)))
    taylor = tf.assign(approx, approx + sign * (x ** tf.cast(n, tf.float64) / tf.exp(tf.lgamma(tf.cast(n, tf.float64) + 1))))
    init = tf.global_variables_initializer()

with tf.Session(graph=taylorGraph) as sess:
    init.run()
    for i in range(10):
        next_n.eval()
        next_sign.eval()
        print(taylor.eval())
    print(approx.eval())
