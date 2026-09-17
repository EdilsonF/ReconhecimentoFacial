# API de Reconhecimento Facial (DeepFace + FastAPI)

Esta é uma API robusta construída em Python utilizando o framework **FastAPI** e a biblioteca de inteligência artificial **DeepFace**. Ela foi desenhada para integração direta com aplicações web (especificamente **Oracle APEX**), permitindo verificação (1:1) e reconhecimento facial em tempo real (1:N).

## Funcionalidades Principais

* **Verificação 1:1 (`/verify`):** Compara duas imagens para validar se são a mesma pessoa.
* **Reconhecimento 1:N em Tempo Real (`/recognize`):** Recebe apenas uma imagem da câmera, procura por um rosto compatível na base de dados (`database/`) utilizando o modelo **Facenet512**, e retorna as coordenadas (bounding box) e a matrícula.
* **Cadastro de Novos Rostos (`/nova-facial`):** Recebe a matrícula e o Base64 da nova foto, salva no disco, e destrói o cache local para garantir reconhecimento imediato do novo rosto.
* **Padding Auto-Correction:** Conserta paddings de Base64 corrompidos automaticamente.

## Stack Tecnológica

* **Python 3**
* **FastAPI** & **Uvicorn**
* **DeepFace** (Modelo: Facenet512) & **OpenCV**
* **Ngrok** (Para bypass de proxy corporativo)

## Como Executar

1. **Instale as dependências:**
   ```bash
   pip install fastapi uvicorn deepface opencv-python pydantic numpy
   ```

2. **Inicie o servidor:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
