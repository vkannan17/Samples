import tensorflow as tf
v1 = tf.Variable(1, name="v1")
v2 = tf.Variable(-1, name="v2")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, "temp\model.ckpt")
print("Model saved in file: %s" % save_path)
