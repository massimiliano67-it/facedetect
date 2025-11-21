import uvicorn
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import io
import base64
import json
import uuid
import cv2
import tempfile
import shutil
import os
# ... (el resto de tus imports: FastAPI, torch, etc)
from typing import List, Optional

# --- INICIALIZACIÓN ---
app = FastAPI()

# Configuración CORS (Permite conectar React con Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado del procesamiento de video para consultar desde el frontend
VIDEO_STATUS = {
    "processing": False,
    "filename": "",
    "frames_processed": 0,
    "faces_found": 0
}


# --- CARGA DE MODELOS DE IA ---
# Se descargan automáticamente la primera vez
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Corriendo IA en: {device}")

#mtcnn = MTCNN(keep_all=True, device=device, margin=0, min_face_size=20)
# min_face_size=10 ayuda a detectar caras más lejanas en fotos grupales
mtcnn = MTCNN(keep_all=True, device=device, margin=0, min_face_size=10)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- BASE DE DATOS EN MEMORIA ---
# Almacena todas las caras procesadas desde que se inició el servidor
ALL_FACES_DB = [] 
# Mapeo de IDs de cluster a Nombres (ej: "0" -> "Juan Perez")
CLUSTER_NAMES = {}

class DeleteFaceRequest(BaseModel):
    face_id: str

class DeleteClusterRequest(BaseModel):
    cluster_id: str

# --- MODELOS DE DATOS (PYDANTIC) ---
# Definimos esto ANTES de los endpoints para evitar el error 422
class RenameClusterRequest(BaseModel):
    cluster_id: str
    new_name: str

class MoveFaceRequest(BaseModel):
    face_id: str
    to_cluster_id: str
    new_cluster_name: Optional[str] = None # Opcional: por si queremos crear grupo nuevo

# --- FUNCIONES AUXILIARES ---

def image_to_base64(img_pil):
    """Convierte imagen PIL a string Base64 para enviar al frontend"""
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_db_local():
    """Guarda una copia de seguridad simple en JSON (sin las imágenes b64 para no pesar)"""
    simple_data = []
    for face in ALL_FACES_DB:
        simple_data.append({
            "id": face["id"],
            "filename": face["filename"],
            "manual_cluster": face["manual_cluster"]
        })
    with open("backup_db.json", "w") as f:
        json.dump({"faces": simple_data, "names": CLUSTER_NAMES}, f, indent=4)

def build_response():
    """
    Construye la respuesta JSON calculando el % de confianza basado en
    la distancia de cada cara al centro (promedio) de su cluster.
    """
    if not ALL_FACES_DB:
        return {}

    # 1. Obtener embeddings y ejecutar DBSCAN
    all_embeddings = [f["embedding"] for f in ALL_FACES_DB]
    # Usamos metric='euclidean' para que las distancias sean coherentes
    clustering = DBSCAN(eps=0.75, min_samples=1, metric='euclidean').fit(all_embeddings)
    labels = clustering.labels_

    # 2. Agrupación Temporal para cálculos matemáticos
    # Primero organizamos todo en grupos para poder calcular los centros
    temp_grouped = {}
    
    for i, label in enumerate(labels):
        face_data = ALL_FACES_DB[i]
        
        # Determinar ID final (Manual mata a IA)
        if face_data["manual_cluster"] is not None:
            final_cluster_id = face_data["manual_cluster"]
        else:
            final_cluster_id = str(label)

        if final_cluster_id not in temp_grouped:
            temp_grouped[final_cluster_id] = []
        
        temp_grouped[final_cluster_id].append(face_data)

    # 3. Calcular Centroides y Confianza, y armar respuesta final
    final_response = {}

    for cluster_id, faces_in_group in temp_grouped.items():
        # Obtener nombre
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Persona {cluster_id}")
        
        final_response[cluster_id] = {
            "id": cluster_id,
            "name": cluster_name,
            "faces": []
        }

        # -- MATEMÁTICA DE CONFIANZA --
        # Calculamos el vector promedio (la "cara ideal" de este grupo)
        group_embeddings = [f["embedding"] for f in faces_in_group]
        if len(group_embeddings) > 0:
            centroid = np.mean(group_embeddings, axis=0)
        else:
            centroid = None

        for face in faces_in_group:
            # Calcular confianza
            confidence_score = 100 # Por defecto si es foto única
            
            if centroid is not None and len(faces_in_group) > 1:
                # Distancia Euclidiana de esta cara al centro del grupo
                dist = np.linalg.norm(face["embedding"] - centroid)
                
                # Conversión heurística a porcentaje:
                # FaceNet suele tener distancias entre 0.0 y 1.5.
                # 0.0 = Idéntico (100%)
                # 1.0 = Muy diferente (0%)
                # La fórmula: (1 - dist) * 100. Si da negativo, ponemos 0.
                score = (1.0 - (dist * 0.8)) * 100 # El 0.8 es un factor de ajuste
                confidence_score = max(0, min(100, score))
            
            final_response[cluster_id]["faces"].append({
                "id": face["id"],
                "filename": face["filename"],
                "image": face["image"],
                "confidence": round(confidence_score, 1) # <--- NUEVO CAMPO
            })

    return final_response


def get_all_embeddings_by_cluster():
    """
    Devuelve un diccionario con TODOS los embeddings organizados por cluster.
    Estructura: { "cluster_id": [vector1, vector2, vector3...] }
    """
    if not ALL_FACES_DB:
        return {}
    
    # 1. Simulamos un clustering rápido para saber el estado actual
    all_embeddings = [f["embedding"] for f in ALL_FACES_DB]
    clustering = DBSCAN(eps=0.75, min_samples=1, metric='euclidean').fit(all_embeddings)
    labels = clustering.labels_

    cluster_map = {}

    for i, label in enumerate(labels):
        face = ALL_FACES_DB[i]
        
        # Respetar asignación manual
        if face["manual_cluster"] is not None:
            cid = face["manual_cluster"]
        else:
            cid = str(label)
        
        if cid not in cluster_map:
            cluster_map[cid] = []
        
        # Guardamos el vector puro
        cluster_map[cid].append(face["embedding"])

    return cluster_map

def find_best_match_nearest_neighbor(new_embedding, cluster_map, threshold=0.65):
    """
    Compara la cara nueva contra TODAS las caras de cada cluster.
    Si se parece mucho a CUALQUIERA de las fotos del cluster, es un match.
    
    threshold: 0.65 es un poco más flexible que 0.6 para fotos grupales.
    """
    best_id = None
    global_min_dist = float('inf')

    for cid, embeddings_list in cluster_map.items():
        # Calculamos la distancia contra CADA foto de este grupo
        for known_emb in embeddings_list:
            dist = np.linalg.norm(new_embedding - known_emb)
            
            if dist < global_min_dist:
                global_min_dist = dist
                # Si encontramos una foto muy parecida, guardamos ese ID candidato
                if dist < threshold:
                    best_id = cid

    return best_id, global_min_dist

def find_best_match(new_embedding, centroids, threshold=0.6):
    """
    Compara una cara nueva con los centroides conocidos.
    threshold: Distancia máxima (Menor = más estricto). 0.6 es buen punto para FaceNet.
    """
    best_id = None
    min_dist = float('inf')

    for cid, centroid in centroids.items():
        dist = np.linalg.norm(new_embedding - centroid)
        if dist < min_dist:
            min_dist = dist
            best_id = cid

    if min_dist < threshold:
        return best_id, min_dist
    return None, min_dist

# --- NUEVA LÓGICA PARA VIDEO ---

def process_video_task(temp_file_path, original_filename, frame_skip=24):
    global ALL_FACES_DB, VIDEO_STATUS
    
    VIDEO_STATUS["processing"] = True
    VIDEO_STATUS["filename"] = original_filename
    VIDEO_STATUS["frames_processed"] = 0
    VIDEO_STATUS["faces_found"] = 0

    # 1. Pre-calcular a quién conocemos
    known_centroids = get_known_centroids()

    cap = cv2.VideoCapture(temp_file_path)
    count = 0
    
    print(f"--> Hilo: Iniciando video {original_filename}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 

            if count % frame_skip == 0:
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_rgb)
                    x_aligned = mtcnn(pil_img)

                    if x_aligned is not None:
                        if len(x_aligned.shape) == 3:
                            x_aligned = x_aligned.unsqueeze(0)
                        
                        x_aligned = x_aligned.to(device)
                        embeddings = resnet(x_aligned).detach().cpu()

                        for i, emb in enumerate(embeddings):
                            emb_np = emb.numpy()
                            
                            # --- RECONOCIMIENTO EN VIDEO ---
                            matched_id, _ = find_best_match(emb_np, known_centroids, threshold=0.6)
                            # -------------------------------

                            face_tensor = x_aligned[i].cpu().permute(1, 2, 0).numpy()
                            face_image_np = (face_tensor * 128 + 127.5).astype(np.uint8)
                            face_pil = Image.fromarray(face_image_np)

                            ALL_FACES_DB.append({
                                "id": str(uuid.uuid4()),
                                "embedding": emb_np,
                                "image": image_to_base64(face_pil),
                                "filename": f"{original_filename} (F{count})",
                                "manual_cluster": matched_id # Asignamos si lo conocemos
                            })
                            VIDEO_STATUS["faces_found"] += 1
                except Exception:
                    pass
                
                VIDEO_STATUS["frames_processed"] = count
            count += 1

    except Exception as e:
        print(f"Error video: {e}")
    finally:
        cap.release()
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        save_db_local()
        VIDEO_STATUS["processing"] = False


@app.post("/api/cluster-video")
async def upload_video(
    background_tasks: BackgroundTasks, # <--- Inyección de dependencia mágica de FastAPI
    file: UploadFile = File(...)
):
    if VIDEO_STATUS["processing"]:
        raise HTTPException(status_code=400, detail="Ya hay un video procesándose. Espera a que termine.")

    # 1. Guardar archivo temporal
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            shutil.copyfileobj(file.file, temp_video)
            temp_video_path = temp_video.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando video: {e}")

    # 2. Lanzar la tarea en SEGUNDO PLANO (Thread)
    # FastAPI ejecutará 'process_video_task' después de retornar la respuesta al usuario
    background_tasks.add_task(process_video_task, temp_video_path, file.filename, frame_skip=30)

    # 3. Responder inmediatamente
    return {"message": "Video recibido. Procesando en segundo plano.", "status": "started"}

# --- ENDPOINTS (API) ---

@app.get("/api/video-status")
def get_video_status():
    return VIDEO_STATUS


@app.get("/api/clusters")
def get_all_clusters():
    return build_response()

@app.post("/api/cluster-video")
async def upload_video(
    background_tasks: BackgroundTasks, # <--- Inyección de dependencia mágica de FastAPI
    file: UploadFile = File(...)
):
    if VIDEO_STATUS["processing"]:
        raise HTTPException(status_code=400, detail="Ya hay un video procesándose. Espera a que termine.")

    # 1. Guardar archivo temporal
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            shutil.copyfileobj(file.file, temp_video)
            temp_video_path = temp_video.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando video: {e}")

    # 2. Lanzar la tarea en SEGUNDO PLANO (Thread)
    # FastAPI ejecutará 'process_video_task' después de retornar la respuesta al usuario
    background_tasks.add_task(process_video_task, temp_video_path, file.filename, frame_skip=30)

    # 3. Responder inmediatamente
    return {"message": "Video recibido. Procesando en segundo plano.", "status": "started"}


@app.post("/api/cluster-faces")
async def upload_and_cluster(files: List[UploadFile] = File(...)):
    global ALL_FACES_DB
    
    print(f"--- Procesando lote de {len(files)} archivos ---")

    # 1. Obtener mapa completo de caras conocidas
    known_clusters_map = get_all_embeddings_by_cluster()

    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            x_aligned = mtcnn(img)

            if x_aligned is not None:
                # Manejo de dimensiones
                if len(x_aligned.shape) == 3:
                    x_aligned = x_aligned.unsqueeze(0)
                
                x_aligned = x_aligned.to(device)
                embeddings = resnet(x_aligned).detach().cpu()

                # Iteramos por cada cara encontrada en la foto
                for i, emb in enumerate(embeddings):
                    emb_np = emb.numpy()
                    
                    # --- NUEVA LÓGICA: VECINO MÁS CERCANO ---
                    # Subimos un poco el threshold a 0.65 o 0.7 para tolerar fotos grupales
                    # Si ves falsos positivos, bájalo a 0.6
                    matched_id, dist = find_best_match_nearest_neighbor(
                        emb_np, 
                        known_clusters_map, 
                        threshold=0.65 
                    )
                    
                    manual_id = None
                    if matched_id:
                        cluster_name = CLUSTER_NAMES.get(matched_id, f"ID {matched_id}")
                        print(f"✅ MATCH ENCONTRADO: {file.filename} (Cara {i}) -> {cluster_name} (Dist: {dist:.4f})")
                        manual_id = matched_id
                    else:
                        print(f"⚠️ NO MATCH: {file.filename} (Cara {i}) -> Distancia más cercana: {dist:.4f}")
                    # ----------------------------------------

                    face_tensor = x_aligned[i].cpu().permute(1, 2, 0).numpy()
                    face_image_np = (face_tensor * 128 + 127.5).astype(np.uint8)
                    face_pil = Image.fromarray(face_image_np)

                    ALL_FACES_DB.append({
                        "id": str(uuid.uuid4()),
                        "embedding": emb_np,
                        "image": image_to_base64(face_pil),
                        "filename": file.filename,
                        "manual_cluster": manual_id
                    })
                    
                    # Actualizamos el mapa en tiempo real para que si hay 2 fotos de Juan 
                    # en este mismo lote, la segunda reconozca a la primera
                    if manual_id:
                         if manual_id not in known_clusters_map: known_clusters_map[manual_id] = []
                         known_clusters_map[manual_id].append(emb_np)

        except Exception as e:
            print(f"Error en {file.filename}: {e}")

    save_db_local()
    return build_response()

@app.post("/api/rename-cluster")
def rename_cluster(req: RenameClusterRequest):
    # Actualizar diccionario de nombres
    CLUSTER_NAMES[req.cluster_id] = req.new_name
    save_db_local()
    return build_response()

@app.post("/api/move-face")
def move_face(req: MoveFaceRequest):
    # Buscar la cara por ID y forzar su nuevo cluster
    found = False
    for face in ALL_FACES_DB:
        if face["id"] == req.face_id:
            face["manual_cluster"] = req.to_cluster_id
            found = True
            
            # Si mandan nombre nuevo (para crear grupo), lo registramos
            if req.new_cluster_name:
                CLUSTER_NAMES[req.to_cluster_id] = req.new_cluster_name
            break
    
    if not found:
        raise HTTPException(status_code=404, detail="Cara no encontrada")

    save_db_local()
    return build_response()

@app.post("/api/delete-face")
def delete_face(req: DeleteFaceRequest):
    global ALL_FACES_DB
    
    # Filtramos la lista manteniendo solo los que NO coincidan con el ID a borrar
    original_len = len(ALL_FACES_DB)
    ALL_FACES_DB = [f for f in ALL_FACES_DB if f["id"] != req.face_id]
    
    if len(ALL_FACES_DB) == original_len:
         raise HTTPException(status_code=404, detail="Imagen no encontrada")

    save_db_local()
    return build_response()

@app.post("/api/delete-cluster")
def delete_cluster(req: DeleteClusterRequest):
    global ALL_FACES_DB, CLUSTER_NAMES

    # 1. Obtenemos el estado actual para saber qué caras pertenecen a este cluster
    # (Ya que DBSCAN puede cambiar dinámicamente, necesitamos la "foto" actual)
    current_structure = build_response()
    
    if req.cluster_id not in current_structure:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")
    
    # 2. Identificar los IDs de las caras que están en ese cluster
    faces_to_delete = [f["id"] for f in current_structure[req.cluster_id]["faces"]]
    
    # 3. Filtrar la base de datos global eliminando esas caras
    ALL_FACES_DB = [f for f in ALL_FACES_DB if f["id"] not in faces_to_delete]

    # 4. Eliminar el nombre personalizado si existe
    if req.cluster_id in CLUSTER_NAMES:
        del CLUSTER_NAMES[req.cluster_id]

    save_db_local()
    
    # Retornamos la nueva estructura limpia
    return build_response() 


if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(app, host="0.0.0.0", port=8000)