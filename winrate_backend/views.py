from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny

from django.shortcuts import get_object_or_404

from winrate_backend.models import GameData
from winrate_backend.serializers import GameDataSerializer, GameDataListSerializer

from prediction_model.mainCrawler import GameResultCrawler

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GameDataViewSet(viewsets.ModelViewSet):
    queryset = GameData.objects.all()
    serializer_class = GameDataSerializer
    permission_classes = [AllowAny]

    def list(self, request):
        serializer = GameDataListSerializer(self.queryset, many=True)
        return Response(serializer.data)

    @action(methods=['GET'], detail=True)
    def get_date(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)
        return Response(serializer.data["latest_update"])
    
    @action(methods=['POST'], detail=True)
    def crawl_history(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)
        
        crawler = GameResultCrawler(serializer.data["game_id"])
        result = crawler.main_crawler()
        if result:
            return Response(status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(methods=['POST'], detail=True)
    def dataset_preprocess(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)
