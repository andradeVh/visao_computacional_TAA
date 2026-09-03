### 1. Criar e ativar ambiente virtual

Linux

```bash
python3 -m venv venv
source venv/bin/activate

```

Windows

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

```

### 2. Instalar dependências

Dentro do venv

```bash
pip install -r requirements.txt

```

### 3. Baixar modelo do mediapipe

```bash
wget [https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

```

### 4. Executar a aplicação

Com o ambiente virtual ativado e o modelo baixado no diretório raiz, execute o script principal para iniciar a captura da webcam:

```bash
python detector_maos.py

```

```

Os passos de 1 a 3 foram preservados e corrigidos ortograficamente a partir do seu arquivo original.

```
