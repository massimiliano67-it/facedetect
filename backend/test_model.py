import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("facenet_model", "facenet_keras.h5")

try:
    print(f"Intentando cargar el modelo desde: {MODEL_PATH}")
    # Usamos safe_mode=False para evitar el error de la capa Lambda
    model = load_model(MODEL_PATH, safe_mode=False)
    print("\n✅ ¡El modelo se cargó exitosamente!")
    print(f"Arquitectura: {model.name}")
    print("---------------------------------")
    print("Ahora puedes iniciar FastAPI.")

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO AL CARGAR EL MODELO")
    print(f"Detalle: {e}")
    print("---------------------------------")
    print("El archivo sigue siendo el problema (corrupto/incompleto) o hay una incompatibilidad de librerías.")