import tensorflow as tf
import os

# Modern CPU's support provide additional low level instructions
# called AVX (advanced vector instructions) but tensorflow default
# distribution is built without these instructions. The following
# command will just supress the annoying warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y")
f = x*x*y + y + 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)
