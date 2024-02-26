from django.urls import path
from .views import *

urlpatterns = [
    path("",home,name="home"),
    path("chat-with-dataset/",chat_datasets,name="chat_datasets"),
    path("generate-dataset-reports/",generate_and_display_profile,name="generate_and_display_profile"),
    path('download-report/', DownloadReportView.as_view(), name='download_report'),
    path('download/<str:model_filename>/', download_model, name='download_model'),
]
