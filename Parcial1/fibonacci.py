import tensorflow as tf

fibonacciGraph = tf.Graph()
with fibonacciGraph.as_default():
    ni_1 = tf.Variable(1)
    ni_2 = tf.Variable(1)
    nx = tf.Variable(0)
    prev_step = tf.assign(nx, ni_2)
    fibonacci = tf.assign(ni_2, ni_1 + ni_2)
    next_step = tf.assign(ni_1, nx)

    init = tf.global_variables_initializer()

with tf.Session(graph=fibonacciGraph) as sess:
    init.run()
    print(x1.eval())
    print(x2.eval())
    for i in range(10):
        prev_step.eval()
        print(fibonacci.eval())
        next_step.eval()
