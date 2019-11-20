from django.db import models

class GameData(models.Model):
    game_id = models.CharField(max_length=25, unique=True)
    latest_update = models.DateTimeField(auto_now=True) # refresh play history

    #encrypted_account = models.CharField(max_length=25, blank=True)
    #history = models.FileField(blank=True)

    inference_team = models.TextField(blank=True)
    inference_enemy = models.TextField(blank=True)
    inference_secondary = models.TextField(blank=True)

    latest_model_trained = models.DateTimeField(blank=True)
