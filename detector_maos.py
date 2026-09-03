import cv2
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectorMaos:
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5, rastreio_confianca=0.5, cor_pontos=(0, 255, 0), cor_linhas=(0, 0, 255), raio_ponto=5):
        """ 
        Função responsável por inicializar a classe DetectorMaos
        :param modo: Modo de detecção (True para detecção contínua, False para detecção única)
        :param max_maos: Número máximo de mãos a serem detectadas
        :param deteccao_confianca: Percentual mínimo de confiança para considerar uma mão detectada. Se for menor que esse valor, a mão será considerada detectada
        :param rastreio_confianca: Percentual da taxa de rastreio para considerar uma mão rastreada. Se for menor que esse valor, o rastreio dos pontos não será realizado
        :param cor_pontos: Cor dos pontos de referência das mãos (BGR)
        :param cor_linhas: Cor das linhas que conectam os pontos de referência das mãos
        :param raio_ponto: Raio padrão dos círculos desenhados nos pontos de referência
        """
        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_linhas = cor_linhas
        self.raio_ponto = raio_ponto
        
        # Inicialização preventiva para evitar AttributeError
        self.resultados = None 
        
        # Conexões oficiais dos pontos da mão (substitui HAND_CONNECTIONS)
        self.conexoes = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Validação do modelo local antes de instanciar a API
        caminho_modelo = "hand_landmarker.task"
        if not os.path.exists(caminho_modelo):
            print(f"Erro: O arquivo de modelo '{caminho_modelo}' não foi encontrado.")
            print("Certifique-se de baixá-lo conforme as instruções no README.md.")
            sys.exit(1)

        # Inicializa o detector da nova API
        base_options = python.BaseOptions(model_asset_path=caminho_modelo)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_maos,
            min_hand_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def encontrar_maos(self, imagem, desenho=True):
        """
        Função responsável por encontrar as mãos na imagem
        :param imagem: Imagem onde as mãos serão detectadas
        :param desenho: Se True, desenha os pontos de referência das mãos na imagem
        :return: Imagem com as mãos detectadas (se desenho=True) e a lista de pontos de referência das mãos
        """
        # Converte a imagem BGR para o formato mp.Image em RGB
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        
        # Processa a imagem para encontrar as mãos
        self.resultados = self.detector.detect(mp_image)
        
        # Se houver mãos detectadas
        if self.resultados.hand_landmarks:
            for pontos in self.resultados.hand_landmarks:
                if desenho:
                    altura, largura, _ = imagem.shape
                    pontos_px = [(int(p.x * largura), int(p.y * altura)) for p in pontos]
                    
                    # Desenha as linhas que conectam os pontos
                    for p1, p2 in self.conexoes:
                        cv2.line(imagem, pontos_px[p1], pontos_px[p2], self.cor_linhas, 2)
                    
                    # Desenha os pontos de referência utilizando self.raio_ponto
                    for pt in pontos_px:
                        cv2.circle(imagem, pt, self.raio_ponto, self.cor_pontos, cv2.FILLED)
        return imagem
            
    def encontrarPontos(self, imagem, mao_num=0, desenho=True, cor_destaque=(255,0,255), pontos_detectados=None):
        """
        Função responsável por encontrar os pontos de referência das mãos na imagem
        :param imagem: Imagem capturada
        :param mao_num: Número da mão a ser analisada (0 para a primeira mão, 1 para a segunda mão, etc.)
        :param desenho: desenhar o(s) ponto(s) em destaque
        :param cor_destaque: Cor de destaque para os pontos específicos solicitados (BGR)
        :param pontos_detectados: Lista de IDs dos pontos a serem destacados. Se None, não destaca pontos extras.
        :return: Lista com pontos detectados
        """
        # Lista com os pontos detectados
        lista_pontos = []
        
        # Verifica se alguma mão foi detectada (Checagem de segurança adicionada com self.resultados)
        if self.resultados and self.resultados.hand_landmarks and len(self.resultados.hand_landmarks) > mao_num:
            # Obter os pontos da mão detectada, não de todas
            mao = self.resultados.hand_landmarks[mao_num]
            
            # Obter as informações de cada ponto da mão
            for id, ponto in enumerate(mao):
                # Obter as dimensões da imagem
                altura, largura, _ = imagem.shape
                centro_X, centro_Y = int(ponto.x * largura), int(ponto.y * altura)
                lista_pontos.append([id, centro_X, centro_Y])
                
                # Desenhar o ponto de destaque na imagem, se solicitado
                if desenho and pontos_detectados is not None:
                    # Verifica se o ID atual está na lista de pontos desejados
                    if id in pontos_detectados:
                        # Utiliza o raio padrão configurado no __init__ + um incremento visual
                        cv2.circle(imagem, (centro_X, centro_Y), self.raio_ponto + 3, cor_destaque, cv2.FILLED)
        return lista_pontos

def main():
    cap = cv2.VideoCapture(0)
    # Inicializa a classe configurando também o raio padrão dos pontos
    detector = DetectorMaos(cor_pontos=(0, 255, 0), cor_linhas=(0, 0, 255), raio_ponto=5)
    
    # Validação do estado da câmera
    while cap.isOpened():
        sucesso, imagem = cap.read()
        if not sucesso:
            print("Falha ao capturar vídeo.")
            break
            
        imagem = cv2.flip(imagem, 1)
        imagem = detector.encontrar_maos(imagem)
        
        # Testando o novo parâmetro: Destacando as pontas do indicador (8) e do polegar (4)
        lista_pontos = detector.encontrarPontos(imagem, mao_num=0, desenho=True, pontos_detectados=[4, 8])
        
        cv2.imshow("Captura", imagem)
        
        # Condição de saída segura ao pressionar a tecla ESC (código 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    # Liberação de recursos após a quebra do laço
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()