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
from prediction_model.secondary_model import secondary_model_train, secondary_model_predict
from prediction_model.inference import inference, average

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

    @action(methods=['GET'], detail=False)
    def search(self, request):
        serializer = GameDataListSerializer(self.queryset, many=True)
        for data in serializer.data:
            if request.query_params["game_id"] == data["game_id"]:
                return Response({"id": data["id"], "game_id": data["game_id"]})
        return Response(status=status.HTTP_404_NOT_FOUND)
    
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

    @action(methods=['GET'], detail=True)
    def model_train(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)
        
        primary_model_train(serializer.data["game_id"], "team")
        primary_model_train(serializer.data["game_id"], "enemy")
        secondary_model_train(serializer.data["game_id"])
        
        '''
        inference_team = inference(serializer.data["game_id"], "team")
        inference_enemy = inference(serializer.data["game_id"], "enemy")
        inference_secondary = inference(serializer.data["game_id"], "secondary")

        dump_team = json.dumps(inference_team)
        dump_enemy = json.dumps(inference_enemy)
        dump_secondary = json.dumps(inference_secondary)

        data = {
            "inference_team": dump_team, 
            "inference_enemy": dump_enemy, 
            "inference_secondary": dump_secondary
            }

        serializer = GameDataSerializer(game_data, data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        '''
        
    @action(methods=['GET'], detail=True)
    def predict(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)

        input_team = process_input(literal_eval(request.query_params["team"]))
        input_enemy = process_input(literal_eval(request.query_params["enemy"]))

        predict_team = primary_model_predict(serializer.data["game_id"], "team", input_team)
        predict_enemy = primary_model_predict(serializer.data["game_id"], "enemy", input_enemy)    

        predict_secondary = secondary_model_predict(serializer.data["game_id"], predict_team, predict_enemy)

        return Response(predict_secondary)

    @action(methods=['GET'], detail=True)
    def inference(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)

        return Response(inference(serializer.data["game_id"], request.query_params["model_type"]))

    @action(methods=['GET'], detail=True)
    def average(self, request, pk):
        game_data = get_object_or_404(self.queryset, pk=pk)
        serializer = GameDataSerializer(game_data)

        game_id = serializer.data["game_id"]
        return Response(average(game_id))