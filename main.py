from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import cv2
import numpy as np
import base64
import logging
import time

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DeepFaceAPI")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API de Reconhecimento Facial",
    description="API para verificar se duas imagens contêm a mesma pessoa usando DeepFace",
    version="1.0.0"
)

# Permitir requisições AJAX diretas do navegador (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    image1_base64: str
    image2_base64: str

class RecognizeRequest(BaseModel):
    image_base64: str

class NovaFacialRequest(BaseModel):
    matricula: str
    image_base64: str

def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decodes a base64 string to a cv2 image (numpy array).
    Handles potential data URI headers like 'data:image/jpeg;base64,'.
    """
    try:
        # Check if the string has the data URI scheme prefix and remove it
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_str)
        
        # Convert bytes to numpy array
        np_arr = np.frombuffer(img_data, np.uint8)
        
        # Decode numpy array to image (cv2)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem.")
            
        return img
    except Exception as e:
        raise ValueError(f"Erro ao processar imagem base64: {str(e)}")

@app.post("/verify")
def verify_faces(request: VerifyRequest):
    logger.info("--- Nova requisição de verificação recebida ---")
    try:
        # 1. Obter as strings base64 limpas (remover 'data:image/...;base64,')
        img1 = request.image1_base64
        img2 = request.image2_base64
        
        logger.info(f"Tamanho Base64 Recebido (Img1 Banco): {len(img1)} caracteres")
        logger.info(f"Tamanho Base64 Recebido (Img2 Webcam): {len(img2)} caracteres")
        
        # Corrigir o padding do base64 se necessário
        def fix_padding(b64_str, image_name):
            padding_needed = 4 - (len(b64_str) % 4)
            if padding_needed and padding_needed != 4:
                b64_str += "=" * padding_needed
                logger.info(f"Padding corrigido em {image_name} (+{padding_needed} caracteres '=' adicionados)")
            return b64_str
            
        if not img1.startswith("data:image"):
            img1 = "data:image/jpeg;base64," + fix_padding(img1, "Img1")
        else:
            prefix, data = img1.split(",", 1)
            img1 = prefix + "," + fix_padding(data, "Img1")
            
        if not img2.startswith("data:image"):
            img2 = "data:image/jpeg;base64," + fix_padding(img2, "Img2")
        else:
            prefix, data = img2.split(",", 1)
            img2 = prefix + "," + fix_padding(data, "Img2")
        
        logger.info("Iniciando verificação no DeepFace.verify()... aguarde.")
        start_time = time.time()
        
        # 2. Executar verificação com DeepFace
        result = DeepFace.verify(
            img1_path=img1, 
            img2_path=img2, 
            enforce_detection=False
        )
        
        elapsed = time.time() - start_time
        logger.info(f"DeepFace.verify() concluído em {elapsed:.2f} segundos!")
        logger.info(f"Resultado bruto do DeepFace: {result}")
        
        # 3. Retornar o resultado esperado
        match = bool(result.get("verified", False))
        distance = float(result.get("distance", 0.0))
        
        logger.info(f"Conclusão: Match={match}, Distância={distance:.4f}")
        logger.info("---------------------------------------------")
        
        return {
            "match": match,
            "distance": distance,
            "model": result.get("model", "VGG-Face")
        }
        
    except ValueError as ve:
        logger.error(f"Erro de decodificação na imagem: {str(ve)}")
        logger.info("---------------------------------------------")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro Interno no DeepFace: {str(e)}", exc_info=True)
        logger.info("---------------------------------------------")
        raise HTTPException(status_code=400, detail=f"Erro na verificação facial: {str(e)}")

@app.post("/recognize")
def recognize_face(request: RecognizeRequest):
    logger.info("--- Nova requisição de reconhecimento 1:N recebida ---")
    try:
        img = request.image_base64
        
        # Corrigir o padding do base64 se necessário
        def fix_padding(b64_str):
            padding_needed = 4 - (len(b64_str) % 4)
            if padding_needed and padding_needed != 4:
                b64_str += "=" * padding_needed
            return b64_str
            
        if not img.startswith("data:image"):
            img = "data:image/jpeg;base64," + fix_padding(img)
        else:
            prefix, data = img.split(",", 1)
            img = prefix + "," + fix_padding(data)
            
        logger.info("Iniciando DeepFace.find()...")
        start_time = time.time()
        
        # Executar busca 1:N no diretório database
        results = DeepFace.find(
            img_path=img, 
            db_path="database",
            model_name="Facenet512",
            enforce_detection=False,
            silent=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"DeepFace.find() concluído em {elapsed:.2f} segundos!")
        
        if len(results) > 0 and not results[0].empty:
            df = results[0]
            best_match = df.iloc[0]
            
            import os
            identity = best_match['identity']
            # O nome do arquivo será a matrícula do funcionário (ex: 12345.jpg -> 12345)
            matricula = os.path.splitext(os.path.basename(identity))[0]
            distance = float(best_match['distance'])
            
            # Tentar pegar as coordenadas do rosto
            box = {}
            if 'source_x' in df.columns:
                box = {
                    "x": int(best_match['source_x']),
                    "y": int(best_match['source_y']),
                    "w": int(best_match['source_w']),
                    "h": int(best_match['source_h'])
                }
            
            logger.info(f"Reconhecido: {matricula} com distância {distance:.4f}")
            
            return {
                "recognized": True,
                "matricula": matricula,
                "distance": distance,
                "box": box
            }
        else:
            logger.info("Nenhum match encontrado. Pegando coordenadas do rosto para o frontend...")
            box = None
            try:
                # Extrai apenas para pegar o quadrado do rosto desconhecido
                faces = DeepFace.extract_faces(img_path=img, enforce_detection=True)
                if len(faces) > 0:
                    face_area = faces[0]['facial_area']
                    box = {
                        "x": face_area['x'],
                        "y": face_area['y'],
                        "w": face_area['w'],
                        "h": face_area['h']
                    }
            except ValueError:
                pass # Nenhum rosto na imagem
                
            return {
                "recognized": False,
                "box": box
            }
            
    except Exception as e:
        logger.error(f"Erro no reconhecimento 1:N: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Erro no reconhecimento: {str(e)}")

@app.post("/nova-facial")
def nova_facial(request: NovaFacialRequest):
    logger.info(f"--- Recebendo nova foto para cadastro da matrícula: {request.matricula} ---")
    try:
        import os
        
        # Corrigir padding se necessário
        img = request.image_base64
        def fix_padding(b64_str):
            padding_needed = 4 - (len(b64_str) % 4)
            if padding_needed and padding_needed != 4:
                b64_str += "=" * padding_needed
            return b64_str
            
        if "," in img:
            prefix, img = img.split(",", 1)
            img = fix_padding(img)
        else:
            img = fix_padding(img)
            
        # Decodificar Base64 e salvar no disco na pasta database
        img_array = decode_base64_image(img)
        file_path = os.path.join("database", f"{request.matricula}.jpg")
        cv2.imwrite(file_path, img_array)
        logger.info(f"Foto salva com sucesso fisicamente em: {file_path}")
        
        # Deletar os arquivos .pkl para forçar a reconstrução do cache (índice)
        pkl_files = [
            "representations_facenet512.pkl",
            "representations_vgg_face.pkl"
        ]
        
        deleted_count = 0
        for pkl in pkl_files:
            pkl_path = os.path.join("database", pkl)
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
                deleted_count += 1
                
        logger.info(f"{deleted_count} arquivos de cache (.pkl) deletados. O DeepFace vai ler o novo funcionário no próximo acesso.")
        
        return {"success": True, "message": f"Foto da matrícula {request.matricula} sincronizada na API com sucesso."}
    
    except Exception as e:
        logger.error(f"Erro ao cadastrar nova facial: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Erro ao sincronizar nova foto: {str(e)}")

@app.get("/")
def root():
    return {"message": "API de Reconhecimento Facial está rodando. Acesse /docs para a documentação."}
