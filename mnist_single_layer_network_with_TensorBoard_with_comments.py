# 必要なモジュールをインポート
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 乱数準備
np.random.seed(20160612)
tf.set_random_seed(20160612)

# MNISTのデータセットを用意
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# NNの構成要素を一つのクラスにまとめて定義
class SingleLayerNetwork:
    # インスタンスを作成した時に最初に呼び出されるコンストラクタ
    # 引数:
    #  num_units: 隠れ層のノード数
    def __init__(self, num_units):
        # 次のwith構文でグラフコンテキストを開始
        with tf.Graph().as_default():
            # 各種構成要素の定義
            self.prepare_model(num_units)
            # セッションの用意
            self.prepare_session()

    # NNの構成要素を定義
    # 引数:
    #  num_units: 隠れ層のノード数
    def prepare_model(self, num_units):
        # with構文によって'input'というネームスコープのコンテキストを
        # 設定し入力層をグループ化
        # このグループかによってネットワークグラフ表示の際に同じグループの
        # 要素が1つの枠にまとめて表示される
        # 入力層に含まれるPlaceholderの定義
        with tf.name_scope('input'):
            # nameオプションでネットワークグラフ上に表示する名前(='input')を指定
            x = tf.placeholder(tf.float32, [None, 784], name='input')

        # with構文によって'hidden'というネームスコープのコンテキストを
        # 設定し隠れ層をグループ化
        # 隠れ層に含まれるVariable, 計算値の定義
        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([784, num_units]),
                             name='weights')        
            b1 = tf.Variable(tf.zeros([num_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')
        

        # with構文によって'output'というネームスコープのコンテキストを
        # 設定し出力層をグループ化
        # 出力層に含まれるVariable, 計算値の定義
        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='softmax')

        # 損失関数の定義
        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            # 最適化アルゴリズムの指定
            train_step = tf.train.AdamOptimizer().minimize(loss)

        # 正解率の計算
        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32), name='accuracy')

        # 学習に伴う値の変化をグラフに表示したい要素を宣言
        # 損失, 正解率, パラメータ行列w1, w0, バイアスb1, b0を表示
        # tf.scalar_summaryはスカラー値をとる要素について
        # その変化を折れ線グラフに表示
        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        # tf.histogram_summaryは複数の要素を含む多次元リストについて
        # それらの値の分布をヒストグラムに表示
        tf.histogram_summary("weights_hidden", w1)
        tf.histogram_summary("biases_hidden", b1)
        tf.histogram_summary("weights_output", w0)
        tf.histogram_summary("biases_output", b0)
                
        # クラスの外部から参照する必要のある変数をインスタンス変数として公開
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
            
    # セッションの用意
    def prepare_session(self):
        sess = tf.Session()
        # Variableの初期化
        sess.run(tf.initialize_all_variables())
        # TensorBoardが参照するデータの出力準備
        # prepare_model内で宣言した要素をまとめたサマリーオブジェクトを作成し
        # 変数summaryに格納
        summary = tf.merge_all_summaries()
        # データの出力先ディレクトリを指定してデータ出力用の
        # SummaryWriterオブジェクトを作成した上で変数writerに格納
        writer = tf.train.SummaryWriter("/tmp/mnist_sl_logs", sess.graph)
        
        # インスタンス変数として公開
        # この後,パラメータの最適化処理を行う中でこれらのオブジェクトを用いて
        # TensorBoardが参照するデータの出力を行う
        self.sess = sess
        self.summary = summary
        self.writer = writer

# SingleLayerNetworkクラスのインスタンスを作成し変数nnに格納
nn = SingleLayerNetwork(1024)

# 2,000回パラメータの更新を繰り返す
i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    # 最適化
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    # 100回更新する毎にサマリーオブジェクトの内容と誤差関数,正解率の値を取得
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x:mnist.test.images, nn.t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # 取得したサマリーオブジェクトの内容を,TensorBoardが参照するデータの
        # 出力用のディレクトリに,SummaryWriterオブジェクトを用いて書き出す
        # TensorBoardがグラフを作成するのに必要な情報として
        # 最適化処理の実施回数iも追加
        nn.writer.add_summary(summary, i)
