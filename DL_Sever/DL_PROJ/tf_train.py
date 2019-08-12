from .models import Train_Set
from .models import Parameter
from .models import PredictData

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import utils
#import dataframe as df
# django_content_type

para_num = Parameter.objects.count()
ModelName = Parameter.objects.get(id=para_num).train_model
EPOCHS = Parameter.objects.get(id=para_num).epochs
PCA_NUM = Parameter.objects.get(id=para_num).pca_num
LEARNING_RATE = float(Parameter.objects.get(id=para_num).learning_rate)
BATCH_SIZE = Parameter.objects.get(id=para_num).batch_size
my_cost = Parameter.objects.get(id=para_num).cost
my_optimizer = Parameter.objects.get(id=para_num).optimizer

# EPOCHS = Parameter.objects.get(id=para_num).epochs


# model_name = "Predict_Gross"
# model_name = "Predict_Score"
model_name = ModelName

if model_name == "Predict_Score":
    PREDICT_NUM = 10
    PREDICT_NAME = 'imdb_score'
    MULTIPLE = 1
    OHNUM = 10
elif model_name == "Predict_Gross":
    PREDICT_NUM = 9
    PREDICT_NAME = 'gross'
    MULTIPLE = 1000000
    OHNUM = 1000

# EPOCHS = 10
# PCA_NUM = 3
# LEARNING_RATE = 0.001
# BATCH_SIZE = 500

# train_sql_num = Parameter.objects.count()
#     ts = float(Parameter.objects.get(id=train_sql_num).test_size)
#     rs = Parameter.objects.get(id=train_sql_num).random_state
#     pipe_list = [0, 1]
#     if  Parameter.objects.get(id=train_sql_num).optimizer == "0":
#         pipe_list.remove(0)
#     if Parameter.objects.get(id=train_sql_num).cost == "0":
#         pipe_list.remove(1)

# my_optimizer = "tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)"
# my_cost = "tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))"


def plot_interdependency(df):
    '''
    绘出电影的特征与欲预测对象（gross和score）的相关性, 只选取了其中几个，
    另外需要的话可以按照格式和movie_data.csv的特征名进行添加
    '''
    df.plot(y= 'imdb_score', x ='duration',kind='hexbin',
        gridsize=45, sharex=False, colormap='cubehelix', 
        title='Hexbin of Imdb_Score and Duration')
    plt.savefig("DL_PROJ/static/images/Score_Duration.png")
    plt.close()
    df.plot(y= 'imdb_score', x ='gross',kind='hexbin',
        gridsize=45, sharex=False, colormap='cubehelix', 
        title='Hexbin of Imdb_Score and Gross')
    plt.savefig("DL_PROJ/static/images/Score_Gross.png")
    plt.close()
    df.plot(y= 'imdb_score', x ='budget',kind='hexbin',
            gridsize=35, sharex=False, colormap='cubehelix', 
            title='Hexbin of Imdb_Score and Budget')
    plt.savefig("DL_PROJ/static/images/Score_Budget.png")
    plt.close()
    df.plot(y= 'gross', x ='duration',kind='hexbin',
            gridsize=35, sharex=False, colormap='cubehelix', 
            title='Hexbin of Gross and Duration')
    plt.savefig("DL_PROJ/static/images/Gross_Duration.png")
    plt.close()

def plot_pearason(df_num):
    '''
    绘出各个特征间的皮尔逊相关系数 
    :param df_num: 
    :return: 
    '''
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))
    plt.title('Pearson Correlation of Movie Features')
    # Draw the heatmap using seaborn
    sns.heatmap(df_num.astype(float).corr(), linewidths=0.4, vmax=1.0,
                square=True, cmap="YlGnBu", linecolor='black')
    plt.savefig("DL_PROJ/static/images/Pearson_Correlation.png")
    plt.close()
    # plt.show()


def load_data():
    '''
    载入训练数据并处理，返回特征矩阵x_std和标签向量y_std
    '''
    # 从数据库提取movie_data.csv
    set_num = Train_Set.objects.count()
    train_set = Train_Set.objects.get(id=set_num).train_file
    movie = pd.read_csv(train_set) #'movie_metadata.csv'
    movie = movie.dropna()
    plot_interdependency(movie)
    print(movie.head)
    str_list = [] # Removing coloumns with strings
    for colname, colvalue in movie.iteritems():
        if type(colvalue[1]) == str:
            str_list.append(colname)
    NumListAll = movie.columns.difference(str_list)
    str_list.append(PREDICT_NAME)           
    NumListTrain=movie.columns.drop(str_list)
    NumListOrder = movie[PREDICT_NAME]
    # NumListOrder = NumListOrder.as_matrix()
    # print(NumListAll)
    print(NumListTrain)
    print(NumListOrder)
    movie_num = movie[NumListAll]

    df_num = movie_num.fillna(value=0, axis=1)
    plot_pearason(df_num)

    MovieNumTrain = movie[NumListTrain]
    # # Extracting 'imdb_score' from upload and deleting the coloumn from datase
    #movie_num.head()
    numpyMatrix = movie_num.as_matrix()
    numpyMatrix_train = MovieNumTrain.as_matrix()
    # numpyMatrix_label = NumListOrder.as_matrix()
    print(numpyMatrix_train.shape)
    x_std= numpyMatrix_train
    y_std = numpyMatrix[:, [PREDICT_NUM]] / MULTIPLE

    print(y_std[0])
    print(y_std.shape)

    movie_num = movie_num.fillna(value=0, axis=1) #replacing nan values with 0
    X = movie_num.values # Normalizing our data using sklearn
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    # y_std=X_std[:, [15]]
    print(y_std)

    # Calculating Eigenvectors and eigenvalues of Cov matirx
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] 
    cum_var_exp = np.cumsum(var_exp)

    pca = PCA(n_components=PCA_NUM) #Running PCA on our normalized data to reduce dimensionality from 15 to PCA_NUM
    x_std = pca.fit_transform(X_std)
    return x_std, y_std


def get_one_hot():
    '''
    对标签向量进行one-hot编码
    '''
    xs, ys = load_data()
    # ys = y_std
    # xs = x_std
    print(xs.shape)
    print(ys.shape)
    # First create a tensorflow session
    sess = tf.Session()
    # Now create an operation that will calculate the mean of our images
    mean_xs = tf.reduce_mean(xs, axis=[1], keep_dims=False, name=None, reduction_indices=None)

    mean_xs = sess.run(mean_xs)

    print(mean_xs.shape)
    mean_xs = np.reshape(mean_xs,(3756,1))
    print(mean_xs)

    mean_xs_4d = tf.reduce_mean(xs, reduction_indices=0, keep_dims=True)
    print(mean_xs_4d.get_shape())

    subtraction = xs - mean_xs_4d
    # Now computing the standard deviation by calculating the
    # square root of the expected squared differences
    std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, reduction_indices=0))
    # Now calculate the standard deviation using your session
    std_img = sess.run(std_img_op)

    #Normalising : by substracting from mean and then dividing by standard devitaion
    xs = tf.reshape(xs,[3756,PCA_NUM])
    norm_imgs_op = (xs-mean_xs)
    norm_imgs_op = norm_imgs_op/std_img;
    sess.run(tf.global_variables_initializer());
    norm_imgs = sess.run(norm_imgs_op);
    xs = norm_imgs
    #Converting our imdb values(ys) into one hot encodings of shape (3756,10)
    ys=np.asarray(ys)
    one_hot_y=np.zeros((3756,OHNUM), dtype=float, order='C')
    import math
    for r in range(3756):
        ys[r]= math.ceil(ys[r])
    for r in range(ys.shape[0]):
        i=ys[r]
        i=i[0].astype(int)
        yz=np.zeros((1,OHNUM), dtype=float, order='C')
        yz[0,i-1]=1
        np.vstack((one_hot_y,yz))
    print(one_hot_y.shape)
    print(len(ys))
    return xs, ys, one_hot_y


def split_data():
    '''
    分割数据
    '''
    xs, ys, one_hot_y = get_one_hot()
    #Splitting our upload into training,validation and test set
    x_train = xs[range(2000),:]
    y_train = ys[range(2000),:]
    x_valid = xs[range(2000,3000),:]
    y_valid = one_hot_y[range(2000,3000),:]
    y_valid_pred = ys[range(2000, 3000), :]
    x_test = xs[range(3000,3756),:]
    y_test = one_hot_y[range(3000,3756),:]
    y_test_prd = ys[range(3000,3756),:]
    print(x_train.shape)
    #Creating a computational graph on Tensorflow
    X = tf.placeholder(tf.float32,name="X",shape=[None,PCA_NUM])
    W = tf.Variable(tf.random_normal([PCA_NUM,5],dtype=tf.float32,stddev = 0.1,name ="W"))
    h = tf.matmul(X,W)
    b = tf.Variable(tf.constant([0,1],dtype=tf.float32,shape=[5],name = "b"))
    h = tf.nn.bias_add(h,b)
    h = tf.nn.relu(h)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, X, W, h, b, h, y_valid_pred, y_test_prd



def linear(x, n_output, name=None, activation=None, reuse=None):
    '''
    Linear function that creates a fully connected layer
    :param x: 
    :param n_output: 
    :param name: 
    :param activation: 
    :param reuse: 
    :return: 
    '''
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)
    n_input = x.get_shape().as_list()[1]
    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)
       # h = tf.ceil(h,name=None)
        if activation:
            h = activation(h)
        return h, W

def train_demo():
    '''
    训练主函数
    '''
    x_train, y_train, x_valid, y_valid, x_test, y_test, X, W, h, b, h, y_valid_pred, y_test_prd = split_data()
    print(y_test.shape, y_train.shape)
    tf.reset_default_graph()

    # Create a placeholder of None x 4 dimensions and dtype tf.float32
    # def create_model():
    X = tf.placeholder(tf.float32,name="X",shape=[None,PCA_NUM]);
    Y = tf.placeholder(tf.float32,name="Y",shape=[None,1]);
    labels = tf.placeholder(tf.float32,name="labels",shape=[None,OHNUM]);

    # We'll create 7 hidden layers.  Let's create a variable to say how many neurons we want for each of the layers
    h1, W1 = linear(
        x=X, n_output=64, name='linear1', activation=tf.nn.relu)
    h2, W2 = linear(
        x=h1, n_output=32, name='linear2', activation=tf.nn.relu)
    h3, W3 =  linear(
        x=h2, n_output=10, name='linear3', activation=tf.nn.relu)
    Y_pred, W4 =  linear(
        x=h3, n_output=1, name='linear4', activation=None)
    #h5, W5 =  linear(
       # x=h4, n_output=5, name='linear5', activation=tf.nn.relu)
    #h6,5 W6 =  linear(
       # x=h5, n_output=5, name='linear6', activation=tf.nn.relu)
    #h7, W7 =  linear(
     #   x=h6, n_output=5, name='linear7', activation=tf.nn.relu)
    #keep_prob = tf.placeholder(tf.float32)
    #h_drop = tf.nn.dropout(h5, keep_prob)
    #Y_pred, W8 = linear(h1, 10, activation=None, name='pred')


    #cost = tf.nn.sigmoid_cross_entropy_with_logits(Y_pred,Y)
    # Creating a session and executing out graph by feeding data in mini batches

    with tf.Session() as sess:
        #cost = -tf.reduce_sum(Y * tf.log(Y_pred))
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Y_pred,Y))
        cost =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))
        correct_prediction = tf.equal(Y_pred,Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        learning_rate = LEARNING_RATE
        batch_size=BATCH_SIZE
        prediction=tf.argmax(Y_pred,1)
        prediction =tf.shape(prediction)
      #  print prediction.eval(feed_dict={x: mnist.test.images})
        datapoint_size=2000;
    #    optimizer = tf.train.AdamOptimizer().minimize(cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        print(cost)
        prev_training_accuracy = 0.0
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # 保存模型
        if ModelName == "Predict_Gross":
            saver.save(sess, 'DL_PROJ/my_model/gross/model.ckpt')
        elif ModelName == "Predict_Score":
            saver.save(sess, 'DL_PROJ/my_model/score/model.ckpt')
        errors = []
        for i in range(EPOCHS):
            print("step %d"%(i))
            for r in range(100):
                if datapoint_size == batch_size:
                    batch_start_idx = 0
                else:
                    batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
                    batch_end_idx = batch_start_idx + batch_size
                    xb=np.array(x_train[batch_start_idx:batch_end_idx])
                    yb=np.array(y_train[batch_start_idx:batch_end_idx])
                    # print(correct_prediction.eval(feed_dict={X: xb, Y:yb}))
                    train_accuracy = accuracy.eval(feed_dict={
                            X: xb, Y:yb})
                    # print("step %d, training accuracy %g"%(i, train_accuracy))
                    errors.append(cost.eval({X: x_train, Y:y_train}))
                    optimizer.run(feed_dict={X: x_train, Y:y_train})
                 #   if(correct_prediction.eval(feed_dict={X: xb, Y:yb})==True):
                 #       print(Y);
                    prev_training_cost = train_accuracy
        # print("validation accuracy %g"%accuracy.eval(feed_dict={
        #            X: x_valid, Y:y_valid}))
        # print("test accuracy %g"%accuracy.eval(feed_dict={
        #            X: x_test, Y:y_test}))
        plt.plot([np.mean(errors[i-20:i]) for i in range(len(errors))])
        # plt.show()
        plt.savefig("DL_PROJ/static/images/errors.png")
        plt.close()

        # 自定义accuracy
        pred_targets = []
        pred_var = []
        for i in range(len(x_test)):
            temp_pred = Y_pred.eval(feed_dict={X:np.reshape(x_test[i],[1,PCA_NUM]),Y:np.reshape(y_test_prd[i],[1,1])})
            pred_targets.append(temp_pred[0])
        for i in range(len(pred_targets)):
            # print(i)
            con_value = float(pred_targets[i][0]) - float(y_test_prd[i])
            con_value = con_value ** 2
            pred_var.append(con_value)
        mean_test = 0.
        sum_test = 0.
        mean_var = []
        for i in range(len(y_test_prd)):
            sum_test += y_test_prd[i]
        mean_test = sum_test / len(y_test_prd)
        for i in range(len(y_test_prd)):
            con_value = (y_test_prd[i] - mean_test) ** 2
            mean_var.append(con_value)
        con_std = 1 - (sum(pred_var) / sum(mean_var))
        print(x_test.shape, x_train.shape)
        print(y_test_prd.shape, y_train.shape)
        return con_std
        # return 0

def predict_demo():
    '''
    预测函数，返回预测结果
    '''
    # 取出数据集
    set_num = Train_Set.objects.count()
    train_set = Train_Set.objects.get(id=set_num).train_file
    movie = pd.read_csv(train_set)  # 'movie_metadata.csv'
    pred_num = PredictData.objects.count()
    MovieTitle = PredictData.objects.get(id=pred_num).movie_title + ' ' # 取出要预测电影的名字
    PredType = PredictData.objects.get(id=pred_num).predict_type # 取出要预测的类型
    movie_num = movie[movie.movie_title == MovieTitle].index.tolist() # 索引行数
    print(movie_num)
    pred_num = movie_num[0]
    # 根据要预测的类型选择模型
    if PredType == 'Predict_Gross':
        model_name = 'gross'
    else:
        model_name = 'score'
    meta_path = './DL_PROJ/my_model/' + model_name + '/model.ckpt.meta'
    model_path = './DL_PROJ/my_model/' + model_name

    # 导入图
    saver = tf.train.import_meta_graph(meta_path)
    config = tf.ConfigProto()
    graph = tf.get_default_graph()
    X_placeholder = graph.get_tensor_by_name('X:0')
    Y_placeholder = graph.get_tensor_by_name('Y:0')
    # logits = graph.get_tensor_by_name('softmax_linear/add:0')  # 最终输出结果的tensor
    # config.gpu_options.allow_growth=True
    x_train, y_train, x_valid, y_valid, x_test, y_test, X, W, h, b, h, y_valid_pred, y_test_prd = split_data()

    # 根据行数索引
    if pred_num <= 1999:
        x_predict = x_train
        y_predict = y_train
    elif pred_num > 1999 and pred_num <= 2999:
        pred_num -= 2000
        x_predict = x_valid
        y_predict = y_valid_pred
    else:
        pred_num -= 3000
        x_predict = x_test
        y_predict = y_valid_pred

    # X = tf.placeholder(tf.float32, name="X", shape=[None, PCA_NUM])
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path)) # 导入已保存的模型
        sessOutput = sess.run(Y_placeholder, feed_dict={X_placeholder: np.reshape(x_predict[pred_num], [1, PCA_NUM]),
                                                        Y_placeholder:np.reshape(y_predict[pred_num],[1,1])})
        print(sessOutput)
        return sessOutput[0][0]


