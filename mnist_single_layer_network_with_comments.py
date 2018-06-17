import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNISTのデータセットを取得するためのモジュール
from tensorflow.examples.tutorials.mnist import input_data

# 乱数のシードを設定
np.random.seed(20160612)
# 隠れ層のパラメータを乱数で初期化するためにTensorFlowに対するシードを設定
tf.set_random_seed(20160612)

# インポートしたモジュールを用いてMNISTのデータセットをダウンロードして
# オブジェクトに格納
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 隠れ層のノード数を変数num_unitsに設定
num_units = 1024

### 機械学習ステップ1 ###

# x: 画像のピクセル値(28×28=784)を(後で)入力するPlaceholder 
#    tf.placeholderクラスのインスタンスとして定義
#
# 第1引数(=tf.float32)は行列の各要素の数値の型
# 第2引数(=[None, 784])は行列のサイズを指定している
#   None: Noneを指定することで任意の数のデータを入力可能となる
#   784: 画像1枚のピクセル数
x = tf.placeholder(tf.float32, [None, 784])

# 隠れ層の出力hidden1をどのように計算するかを定義
# w1: パラメータ行列(±2σの切断正規分布(平均0,標準偏差1)からの乱数で初期化)
# b1: バイアス項(0で初期化)
#   w1, b1はともに最適化の対象なのでtf.Variableクラスのインスタンスとして定義
w1 = tf.Variable(tf.truncated_normal([784, num_units]))
b1 = tf.Variable(tf.zeros([num_units]))
# NNへの入力(画像のピクセル値)にパラメータ行列w1を掛けてバイアス項b1を加算,
# 活性化関数reluを適用
# hidden1のサイズ: データ数×num_units(=1024)
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

# 隠れ層の出力からソフトマックス関数を用いて確率を計算する
# w0: パラメータ行列, b0: バイアス項
w0 = tf.Variable(tf.zeros([num_units, 10]))
b0 = tf.Variable(tf.zeros([10]))
# 隠れ層の出力hidden1にパラメータ行列w0を掛けてバイアス項b0を加算,
# ソフトマックス関数を適用
# pのサイズ: データ数×10(ラベルの数)
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)


### 機械学習ステップ2 ###

# t: トレーニングセットの正解ラベルを(後で)格納するPlaceholder
t = tf.placeholder(tf.float32, [None, 10])

# 損失関数の定義
# tf.reduce_sumはベクトルの各成分の和を返す
loss = -tf.reduce_sum(t * tf.log(p))

### 機械学習ステップ3(の一部) ###
# 最適化アルゴリズムtf.train.AdamOptimizerを用いてlossを最小化するという指定
# train_stepをセッション内で評価することによって実際に最適化が行われる
train_step = tf.train.AdamOptimizer().minimize(loss)
### 機械学習ステップ3(の一部)おしまい ###

# テストセットに対する正解率を計算するための準備
# tf.argmaxは複数の要素が並んだリストから最大値をもつ要素のインデックスを取り出す関数
#  第1引数: リストの名前
#  第2引数: どの軸に沿って検索するか(0=>縦方向(列), 1=>横方向(行))
# tf.equalは第1引数と第2引数が一致 => true, otherwise => false
#
# p(各ラベルの確率が入っている)の内,最大の確率値をもつインデックスと
# t(正解ラベル)の1が立っているインデックスが一致しているかを見ている
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))

# tf.castはbool値を1,0に変換
# tf.reduce_meanは全体の平均値(=正解率)を返す
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# セッションを1つ用意して変数sessに格納
# (一般には複数のセッションを定義して各セッションで個別に計算を行うことも可能)
sess = tf.Session()
# tf.Variableで定義したパラメータを全て初期化
sess.run(tf.initialize_all_variables())

### 機械学習ステップ3(最適化) ###
i = 0
for _ in range(2000):
    i += 1

    # 変数mnistに格納されたオブジェクトのメソッドを利用してデータを取り出す
    # 次の指示はトレーニングセットから100個分のデータ(画像とそれに対応するラベル100ずつ)を取り出す
    # ここでは,ミニバッチサイズ=100ということになる
    # mnist.train.next_batchはデータをどこまで取り出したかを記憶していて呼び出す毎に次のデータを取り出す
    # データを最後まで取り出すと再び最初に戻ってデータを返す
    # 参考(https://github.com/tensorflow/tensorflow/blob/v0.6.0/tensorflow/examples/tutorials/mnist/input_data.py)
    # batch_xs: 画像
    # batch_ts: 数字のラベル(1hotベクトルで表現)
    #   ex.) 数字7なら(0,0,0,0,0,0,0,1,0,0)で表現される
    batch_xs, batch_ts = mnist.train.next_batch(100)

    # 次の1行でTensorFlowがパラメータの最適化を行う
    # ステップ1の段階でPlaceholderとして定義していたxとtに実際の値を入力
    # x <= 画像, t <= 正解ラベル
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})

    # 100回パラメータの更新を行う毎にテストセットを用いて評価する
    if i % 100 == 0:
        # lossと正解率をテストセットに対して評価
        loss_val, acc_val = sess.run([loss, accuracy],
            feed_dict={x:mnist.test.images, t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

# 得られた結果を実際の画像で確認
# 0 ~ 9の各々に対してテストセットから正解/不正解の画像を3枚ずつ取り出して表示
# 各画像の上に<予測ラベル>/<正解ラベル>を示す
images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x:images, t: labels}) 

fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for (image, label, pred) in zip(images, labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c < 4 and i == actual) or (c >= 4 and i != actual):
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' % (prediction, actual))
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                           cmap=plt.cm.gray_r, interpolation="nearest")
            c += 1
            if c > 6:
                break
