from django.shortcuts import render, redirect, HttpResponse
from functools import wraps
from django.http import HttpResponseRedirect
from .models import User
from .models import Train_Set
from .models import PredictData
from .models import Parameter
from .tf_train import train_demo, predict_demo

from django import forms

#para_list default value
para_list = [
    {"name":"model", "value":"Predict Gross"},
    {"name":"optimizer", "value":"tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)"},
    {"name":"cost", "value":"tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))"},
    {"name":"learning_rate", "value":"0.001"},
    {"name":"batch_size", "value":"500"},
    {"name":"epochs", "value":"20"},
    {"name":"pca_num", "value":"3"},
]

pipe_para_list = ["optimizer", "cost"]

# 说明：这个装饰器的作用，就是在每个视图函数被调用时，都验证下有没法有登录，
# 如果有过登录，则可以执行新的视图函数，
# 否则没有登录则自动跳转到登录页面。
def check_login(f):
    @wraps(f)
    def inner(request, *arg, **kwargs):
        if request.session.get('is_login') == '1':
            return f(request, *arg, **kwargs)
        else:
            return redirect('/login/')

    return inner


def login(request):
    # 如果是POST请求，则说明是点击登录按扭 FORM表单跳转到此的，那么就要验证密码，并进行保存session
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = User.objects.filter(username=username, password=password)
        print(user)
        if user:
            # 登录成功
            # 1，生成特殊字符串
            # 2，这个字符串当成key，此key在数据库的session表（在数据库存中一个表名是session的表）中对应一个value
            # 3，在响应中,用cookies保存这个key ,(即向浏览器写一个cookie,此cookies的值即是这个key特殊字符）
            request.session['is_login'] = '1'  # 这个session是用于后面访问每个页面（即调用每个视图函数时要用到，即判断是否已经登录，用此判断）
            # request.session['username']=username  # 这个要存储的session是用于后面，每个页面上要显示出来，登录状态的用户名用。
            request.session['user_id'] = user[0].id
            return redirect('/index/')
    # 如果是GET请求，就说明是用户刚开始登录，使用URL直接进入登录页面的
    return render(request, 'login.html')


# @check_login
# def index(request):
#     user_id1 = request.session.get('user_id')
#     # 使用user_id去数据库中找到对应的user信息
#     userobj = User.objects.filter(id=user_id1)
#     print(userobj)
#     return render(request, 'index.html',
#                   {"data": para_list})

@check_login
def get_para(request):
    return render(request, 'index.html', {"data": para_list})

@check_login
def upload_file(request):
    if request.method == 'POST':
        file_obj = request.FILES.get('upload_file')
        o_train_set = Train_Set()
        o_train_set.train_file = file_obj
        o_train_set.save()
        print(file_obj.name, file_obj.size)
        import os
        f = open(os.path.join('static', 'upload', file_obj.name), 'wb')
        for line in file_obj.chunks():
            f.write(line)
        f.close()
        return HttpResponseRedirect('/index')
    elif request.method == 'GET':
        return render(request, 'upload_file.html')

@check_login
def train_model(request):
    accuracy = train_demo()
    # accuracy = load_model()
    para_list = []
    tmp_model = {"name": "accuracy", "value": accuracy}
    para_list.append(tmp_model)
    return render(request, 'train_result.html', {"data":para_list})


@check_login
def predict_index(request):
    return render(request, 'predict.html', {"data": para_list})

@check_login
def predict_model(request):
    if request.method == 'POST':
        if 'predict_movie' in request.POST:
            movie_title = request.POST.get("movie_name", "")
            predict_type = request.POST.get("predict_type", "")
            o_input_file = PredictData()
            o_input_file.movie_title = movie_title
            o_input_file.predict_type = predict_type
            # o_input_file.tag = 0
            o_input_file.save()
            predict_tag = predict_demo()
            para_list = []
            predict_msg = {"title": movie_title, "type":predict_type, "value":predict_tag}
            para_list.append(predict_msg)
            return render(request, 'predict_result.html', {"data":para_list})

@check_login
def index(request):
    user_id1 = request.session.get('user_id')
    # 使用user_id去数据库中找到对应的user信息
    userobj = User.objects.filter(id=user_id1)
    print(userobj)
    if request.method == 'POST':
        if 'update' in request.POST:
            pipeline_list = request.POST.getlist("check_box_list")
            train_model = request.POST.get('train_model', '')
            learning_rate = request.POST.get('learning_rate', '')
            batch_size = request.POST.get('batch_size', '')
            epochs = request.POST.get('epochs', '')
            pca_num = request.POST.get('pca_num', '')


            o_parameter = Parameter()
            o_parameter.train_model = train_model
            tmp_model = {"name": "model", "value": train_model}

            if learning_rate != '':
                o_parameter.learning_rate = learning_rate
                tmp_lr = {"name": "learning_rate", "value": learning_rate}
            else:
                o_parameter.learning_rate = 0.001
                tmp_lr = {"name": "learning_rate", "value": 0.001}
            if batch_size != '':
                o_parameter.batch_size = batch_size
                tmp_bs = {"name": "batch_size", "value": batch_size}
            else:
                o_parameter.random_state = 500
                tmp_bs = {"name": "batch_size", "value": 500}
            if epochs != '':
                o_parameter.epochs = epochs
                tmp_e = {"name": "epochs", "value": epochs}
            else:
                o_parameter.random_state = 20
                tmp_e = {"name": "epochs", "value": 20}
            if pca_num != '':
                o_parameter.pca_num = pca_num
                tmp_pn = {"name": "pca_num", "value": pca_num}
            else:
                o_parameter.pca_num = 3
                tmp_pn = {"name": "pca_num", "value": 3}
            # save Parameter
            o_parameter.save()

            para_list = []
            para_list.append(tmp_model)
            # read value from dataframe
            for pipe_para in ["optimizer", "cost"]:
                if pipe_para == "optimizer":
                    if pipe_para in pipeline_list:
                        if request.POST.get('optimizer_value', '') == '':
                            o_parameter.optimizer = "tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)"
                        else:
                            o_parameter.optimizer = request.POST.get('optimizer_value', '')
                        tmp_pipe = {"name": pipe_para, "value": o_parameter.optimizer }
                    else:
                        tmp_pipe = {"name": pipe_para, "value": "OFF"}
                        o_parameter.optimizer = "tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)"

                elif pipe_para == "cost":
                    if pipe_para in pipeline_list:
                        if request.POST.get('cost_value', '') == '':
                            o_parameter.cost = "tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))"
                        else:
                            o_parameter.cost = request.POST.get('cost_value', '')
                        tmp_pipe = {"name": pipe_para, "value": o_parameter.cost}
                    else:
                        tmp_pipe = {"name": pipe_para, "value": "OFF"}
                        o_parameter.cost = "tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_pred, Y))))"

                else:
                    return HttpResponse('<h2>Error！Please Input Again</h2>')
                para_list.append(tmp_pipe)
            para_list.append(tmp_lr)
            para_list.append(tmp_bs)
            para_list.append(tmp_e)
            para_list.append(tmp_pn)
            # save Parameter
            o_parameter.save()

            return render(request, 'index.html', {"data":para_list})

        elif 'train' in request.POST:
            return HttpResponseRedirect('/train')

        elif 'predict' in request.POST:
            return HttpResponseRedirect('/predict')

        else:
            return render(request, 'index.html')

