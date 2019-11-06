from django.db import models

class GameData(models.Model):
    game_id = models.CharField(max_length=25, unique=True)
    latest_update = models.DateTimeField(auto_now=True) # refresh play history

    #encrypted_account = models.CharField(max_length=25, blank=True)
    #history = models.FileField(blank=True)

    primary_team_inference = models.CharField(max_length=200, blank=True)
    primary_enemy_inference = models.CharField(max_length=200, blank=True)
    primary_team_inference = models.CharField(max_length=200, blank=True)

    latest_model_trained = models.DateTimeField(blank=True)
    