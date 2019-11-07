from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny

from django.shortcuts import get_object_or_404

from winrate_backend.models import GameData
from winrate_backend.serializers import GameDataSerializer, GameDataListSerializer

from prediction_model.mainCrawler import GameResultCrawler
from prediction_model.dataset_preprocessor import DatasetPreprocessor, process_input
from prediction_model.primary_models import primary_model_train, primary_model_predict
from prediction_model.secondary_dataset import secondary_dataset_generate, average_dataset

from ast import literal_eval
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GameDataViewSet(viewsets.ModelViewSet):
    queryset = GameData.objects.all()
    serializer_class = GameDataSerializer
    permission_classes = [AllowAny]

    def list(self, request):
        serializer = GameDataListSerializer(self.queryset, many=True)
        return Response(serializer.data)
    
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

        primary_preprocessor = DatasetPreprocessor(serializer.data["game_id"])
        primary_preprocessor.process_dataset()

        secondary_dataset_generate(serializer.data["game_id"], "team")
        secondary_dataset_generate(serializer.data["game_id"], "enemy")
        average_dataset(serializer.data["game_id"])

        return Response(status=status.HTTP_200_OK)

    @action(methods=['POST'], detail=True)
    def model_train(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)

        if request.data["model_type"] == "primary":
            primary_model_train(serializer.data["game_id"], "team")
            primary_model_train(serializer.data["game_id"], "enemy")
        else:
            pass
        
        return Response(status=status.HTTP_200_OK)

    @action(methods=['GET'], detail=True)
    def predict(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)

        input_team = process_input(literal_eval(request.data["team"]))
        input_enemy = process_input(literal_eval(request.data["enemy"]))

        predict_team = primary_model_predict(serializer.data["game_id"], "team", input_team)
        predict_enemy = primary_model_predict(serializer.data["game_id"], "enemy", input_enemy)
        

        return Response({"team_predict": predict_team})

