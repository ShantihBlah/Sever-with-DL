# Generated by Django 2.0.7 on 2019-03-09 20:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DL_PROJ', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='role',
        ),
    ]