# Generated by Django 2.2.5 on 2019-11-20 13:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('winrate_backend', '0005_auto_20191108_2007'),
    ]

    operations = [
        migrations.CreateModel(
            name='Splash',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(default='media\\default.jpg', upload_to='')),
            ],
        ),
    ]