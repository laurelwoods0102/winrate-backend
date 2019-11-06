from rest_framework import serializers

from winrate_backend.models import GameData

class GameDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameData
        fields = '__all__'
        ordering = ['latest_update']

class GameDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameData
        fields = ['id', 'game_id', 'latest_update']
        ordering = ['latest_update']