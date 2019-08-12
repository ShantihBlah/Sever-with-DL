from django.db import models

# Create your models here.
class User(models.Model):
    username = models.IntegerField(blank=True, null=True, unique=True)  # id
    password = models.CharField(max_length=50, default=None)

class Train_Set(models.Model):
    train_file = models.FileField(upload_to='static/upload')

class PredictData(models.Model):
    movie_title = models.CharField(max_length=1500, default='Avatar')
    predict_type = models.CharField(max_length=100, default='Predict_Gross')
    # tag = models.IntegerField(default=0)

class Parameter(models.Model):
    train_model = models.CharField(max_length=30, default='Predict_Gross')

    optimizer = models.CharField(max_length=1000,
                                 default="tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)")
    cost = models.CharField(max_length=1000,
                            default="tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))")

    learning_rate = models.DecimalField(max_digits=5, decimal_places=4, default=0.001)
    batch_size = models.IntegerField(default=500)
    epochs = models.IntegerField(default=20)
    pca_num = models.IntegerField(default=3)

    # test_size = models.DecimalField(max_digits=4, decimal_places=2, default=0.25)
    # random_state = models.IntegerField(default=2)
    # # 1标志使用该参数 eg: scale——>MaxAbsScaler()
    # optimizer = models.CharField(max_length=1000, default="tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)")
    # vec_H = models.CharField(max_length=1000, default="HashingVectorizer(norm=None, non_negative=True, token_pattern=TOKENS_ALPHANUMERIC,ngram_range=(1, 2))")
    # int = models.CharField(max_length=1000, default="SparseInteractions(degree=2)")
    # scale = models.CharField(max_length=1000, default="MaxAbsScaler()")
    # clf = models.CharField(max_length=1000, default="OneVsRestClassifier(LogisticRegression())")