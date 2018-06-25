# 必要なモジュールをインポート
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 乱数準備
np.random.seed(20160704)
tf.set_random_seed(20160704)

# MNISTデータセット読み込み
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

### 機械学習ステップ1 ###

# 1層目のフィルタ数(=32)を変数num_filters1に格納
num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 784])
# 画像データを入力するx(placeholder)をtf.nn.conv2dに入力可能な形式に変換
# 画像の枚数×画像サイズ(縦(=28)×横(=28))×チャネル数(=1)
# tf.reshapeの引数の最初の-1はplaceholderに格納されているデータ数に応じて
# 適切なサイズに調整してくれる
x_image = tf.reshape(x, [-1,28,28,1])

# 畳み込み層
# フィルタは最適化の対象なのでtf.Variableとして定義
# フィルタサイズ: 5×5
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,num_filters1],
                                          stddev=0.1))
# 関数tf.nn.conv2dを用いて入力データx_imageに対してフィルタW_conv1を適用
# フィルタは1ピクセル毎に動かし,0-paddingを行う
h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                       strides=[1,1,1,1], padding='SAME')

# ReLUを適用すると負の値は0に置き換えられるが
# ここでは閾値b_conv1を用意しこの閾値以下をReLUで0にする
# b_conv1も最適化の対象になる(初期値は0.1を与えている)
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
# 活性化関数ReLUの適用
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

# プーリング層
# この層は最適化の対象になるパラメータを含まない
# tf.nn.max_poolはksizeオプションで指定されたサイズのブロックを
# stridesオプションで指定された感覚で移動させながら
# ブロック内にあるピクセルの最大値で置き換えていく
# ksize/stridesオプション: [1,dy,dx,1]の形式てdy(縦方向)とdx(横方向)を指定
# 最大値ではなく平均値で置き換えるtf.nn.avg_poolも用意されている
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')

# 2層目のフィルタ数(=64)を変数num_filters2に格納
num_filters2 = 64

# 畳み込み層
# フィルタサイズ: 5×5
W_conv2 = tf.Variable(
            tf.truncated_normal([5,5,num_filters1,num_filters2],
                                stddev=0.1))
# 畳み込みフィルタ適用
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                       strides=[1,1,1,1], padding='SAME')


b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
# 活性化関数ReLU適用
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

# プーリング層
# h_pool2: データ数×7×7×64
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')

# 7×7×64の特徴量を1列に並べた1次元のリストに変換
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])

# FC層
num_units1 = 7*7*num_filters2
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

########
# ドロップアウト層
# 変数keep_prob: ドロップアウトにおいて有効な状態を保持するノードの割合
# Placeholderクラスのインスタンスとして定義し,後で値を代入できるようにする
keep_prob = tf.placeholder(tf.float32)
# ドロップアウトしてくれる
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

### 機械学習ステップ2 ###
t = tf.placeholder(tf.float32, [None, 10])
# 損失関数の定義
loss = -tf.reduce_sum(t * tf.log(p))

### 機械学習ステップ3(の一部) ###
# 最適化アルゴリズムの指定
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
### 機械学習ステップ3(の一部)おしまい ###

# 正解率の計算
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# セッションを作成
sess = tf.Session()
# Variableを初期化
sess.run(tf.initialize_all_variables())

# セッションの状態を保存するためにtf.train.Saverオブジェクトを取得して
# 変数saverに格納
saver = tf.train.Saver()

### 機械学習ステップ3(最適化) ### 
i = 0
# パラメータの更新を20,000回繰り返す
for _ in range(20000):
    i += 1
    # ミニバッチサイズ: 50
    batch_xs, batch_ts = mnist.train.next_batch(50)
    # ドロップアウトの有効なノードの割合も忘れずfeed_dictする
    sess.run(train_step,
             feed_dict={x:batch_xs, t:batch_ts, keep_prob:0.5})
    # 500回更新する毎に
    if i % 500 == 0:
        loss_vals, acc_vals = [], []
        # メモリの使用量を減らすためにテストデータを分割して4回に分けて評価
        for c in range(4):
            start = len(mnist.test.labels) / 4 * c
            end = len(mnist.test.labels) / 4 * (c+1)
            loss_val, acc_val = sess.run([loss, accuracy],
                feed_dict={x:mnist.test.images[start:end],
                           t:mnist.test.labels[start:end],
                           keep_prob:1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # saverオブジェクトのsaveメソッドを呼び出してセッションの状態を保存
        # 引数:
        #   保存対象のセッション, 保存用のファイル名, 最適化処理の実施回数
        # <指定したファイル名>-<処理回数>, <指定したファイル名>-<処理回数>.meta
        # 過去5回分のファイルのみが保存されてそれより古いファイルは自動的に削除
        saver.save(sess, 'cnn_session', global_step=i)
