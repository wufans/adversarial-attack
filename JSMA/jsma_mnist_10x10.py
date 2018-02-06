import os

import numpy as np

import matplotlib
#matplotlib.use('Agg')           # noqa: E402 save image without showing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from attacks import jsma


img_size = 28
img_chan = 1
n_classes = 10


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.target = tf.placeholder(tf.int32, (), name='target')
    env.adv_epochs = tf.placeholder_with_default(20, shape=(), name='epochs')
    env.adv_eps = tf.placeholder_with_default(0.2, shape=(), name='eps')
    env.x_jsma = jsma(model, env.x, env.target, eps=env.adv_eps,
                      epochs=env.adv_epochs)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#evaluate(sess, env, X_adv, y_test)
def evaluate(sess, env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(sess, env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        if X_valid is not None:
            evaluate(sess, env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={
            env.x: X_data[start:end],
            env.target: np.random.choice(n_classes)})
        yval[start:end] = y_batch
    #print()
    return yval


def make_jsma(sess, env, X_data, epochs=0.2, eps=1.0, batch_size=128):
    """
    Generate JSMA by running env.x_jsma.
    """
    print('\nMaking adversarials via JSMA')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        feed_dict = {
            env.x: X_data[start:end],
            env.target: np.random.choice(n_classes),
            env.adv_epochs: epochs,
            env.adv_eps: eps}
        adv = sess.run(env.x_jsma, feed_dict=feed_dict)
        X_adv[start:end] = adv
    print()

    return X_adv




#训练部分，可以跳过
print('\nTraining')

train(sess, env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
      name='mnist')

print('\nEvaluating on clean data')
#X_test和Y_test的长度为10000
evaluate(sess, env, X_test, y_test)

print('\nGenerating adversarial data')

X_adv = make_jsma(sess, env, X_test, epochs=30, eps=1.0)
#-------------------------------------
adv_outputs = predict(sess, env, X_adv)
adv_pro = np.zeros(10000)
i = 0
for adv_output in adv_outputs:#得到每一个样本的输出值
    var = np.var(adv_output)
    print("adversarial variance:",var)
    adv_pro[i] = var
    i += 1
'''
for adv_output in adv_outputs:#得到每一个样本的输出值
    #print("adversarial_output",adv_output)
    adv_output.sort()
    sum = 0
    for singal_pro in adv_output:
        sum += singal_pro
    #probability = np.max(adv_output)/sum
    probability = (adv_output[-1]-adv_output[-2])/sum
    print("adversarial probability:",probability)
    adv_pro[i] = probability
    i += 1
'''
#------------------------------------------
print('\nEvaluating on adversarial data')

evaluate(sess, env, X_adv, y_test)
#-------------------------------------------------------------------------------
print('画出正常样本的 1.输出最大概率分布 2.输出概率分布差值')

#print("len--------:",len(predict(sess, env, X_test)))
clean_outputs = predict(sess, env, X_test)
clean_pro = np.zeros(10000)
i = 0
for clean_output in clean_outputs:#得到每一个样本的输出值
    var = np.var(clean_output)
    print("original variance:",var)
    clean_pro[i] = var
    i += 1
'''
for clean_output in clean_outputs:#得到每一个样本的输出值
    print("clean_output",clean_output)
    clean_output.sort()
    sum = 0
    for singal_pro in clean_output:
        sum += singal_pro
    #probability = np.max(clean_output)/sum
    probability = (clean_output[-1]-clean_output[-2])/sum
    print("probability:",probability)
    clean_pro[i] = probability
    i += 1
'''
#-------------------------------------------------------------------------------

print('\nRandomly sample adversarial data from each category')

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(predict(sess, env, X_test), axis=1)
ind = z0 == z1

X_data = X_test[ind]
labels = z0[ind]

X_adv = np.empty((10, 10, 28, 28))

#定义一个概率差值的数组
#adv_pro = []

for source in np.arange(10):
    print('Source label {0}'.format(source))

    X_i = X_data[labels == source]#从样本中取出从1到9的标签图像

    for i, xi in enumerate(X_i):
        found = True
        xi = xi[np.newaxis, :]

        for target in np.arange(10):
            print(' [{0}/{1}] {2} -> {3}'
                  .format(i+1, X_i.shape[0], source, target), end='')

            if source == target:#对应于原图
                xadv = xi.copy()
            else:
                feed_dict = {env.x: xi, env.target: target, env.adv_epochs: 30,
                             env.adv_eps: 1.0}
                xadv = sess.run(env.x_jsma, feed_dict=feed_dict)#生成一个对抗样本

            yadv = predict(sess, env, xadv)
            #-------------------------------------------------------------------
            # sum = 0
            # for singal_pro in yadv[0]:
            #     sum += singal_pro
            # probability = np.max(yadv[0])/sum
            # print("probability:",probability)
            # adv_pro.append(probability)
            #------------------------------------------------
            #print("对抗样本的预测标签值：",yadv)#
            label = np.argmax(yadv.flatten())
            # label_sort = np.sort(yadv)
            # print("最大的值:",label_sort[0][-1])
            # print("第二大的值:",label_sort[0][-2])
            # print("差值：",label_sort[0][-1]-label_sort[0][-2])
            # print("对抗样本和初始样本的概率差值",yadv[0][label]-yadv[0][source])
            #-------------------------------------------------------------------
            found = target == label

            if not found:
                print('Attack Success')#也就是对抗样本预测失败
                break

            X_adv[source, target] = np.squeeze(xadv)
            print(' res: {0} {1:.2f}'.format(label, np.max(yadv)))

        if found:
            break
#adv_pro = np.array(adv_pro)
print('\nGenerating figure')
'''
print("在对抗样本上的最大概率值：",adv_pro)
plt.figure(1,figsize=(10, 10))
#plt.title("Adversarial examples output frequency")
plt.title("Difference of the highest two probabilities in adversarial examples - jsma")
plt.hist(adv_pro)

print("在初始样本上的最大概率值：",clean_pro)
plt.figure(2,figsize=(10, 10))
#plt.title("Original examples output frequency")
plt.title("Difference of the highest two probabilities in original examples - jsma")
plt.hist(clean_pro)
'''
print("在对抗样本上的最大概率：",adv_pro)
plt.figure(1,figsize=(10, 10))
plt.title("Output variance of adversarial examples - jsma")
plt.hist(adv_pro)

print("在初始样本上的最大概率：",clean_pro)
plt.figure(2,figsize=(10, 10))
#plt.title("Original examples output frequency")
plt.title("Output variance of original examples - jsma")
plt.hist(clean_pro)

fig = plt.figure(3,figsize=(10, 10))
gs = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)#调整子图的位置和大小

for i in range(10):
    for j in range(10):
        ax = fig.add_subplot(gs[i, j])
        #print("real label:",i,"perturbed label:",j,"\nadversarial examples",X_adv[i,j])
        ax.imshow(X_adv[i, j], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])

        if i == j:
            for spine in ax.spines:
                ax.spines[spine].set_color('red')#中间的图像框框加粗且为绿色
                ax.spines[spine].set_linewidth(5)

        if ax.is_first_col():
            ax.set_ylabel(i, fontsize=20, rotation='horizontal', ha='right')
        if ax.is_last_row():
            ax.set_xlabel(j, fontsize=20)

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/jsma_mnist_10x10.png')
plt.show()
