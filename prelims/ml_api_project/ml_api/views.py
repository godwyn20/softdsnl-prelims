from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
from django.conf import settings

# Paths to your serialized artifacts
MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_api", "model.pkl")
ENCODER_PATH = os.path.join(settings.BASE_DIR, "ml_api", "label_encoder.pkl")

# Load once at import time
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


class PredictView(APIView):
    """
    POST /api/predict/
    {
      "height": 22.0,
      "width": 6.5,
      "length": 6.5
    }
    → { "prediction": "ceramic" }
    """

    def post(self, request):
        # 1. Extract & validate inputs
        try:
            # fetch as floats
            height = float(request.data.get("height", None))
            width  = float(request.data.get("width",  None))
            length = float(request.data.get("length", None))
        except (TypeError, ValueError):
            return Response(
                {"error": "Please provide numeric values for height, width, and length."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Predict
        try:
            features = [[height, width, length]]
            pred_idx = model.predict(features)
            pred_label = label_encoder.inverse_transform(pred_idx)[0]
        except Exception as e:
            return Response(
                {"error": f"Model prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Return
        return Response({"prediction": pred_label})


class TypesView(APIView):
    """
    GET /api/types/
    → { "types": ["wood", "fabric", "ceramic"] }
    """

    def get(self, request):
        try:
            classes = list(label_encoder.classes_)
            return Response({"types": classes})
        except Exception as e:
            return Response(
                {"error": f"Could not retrieve classes: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
