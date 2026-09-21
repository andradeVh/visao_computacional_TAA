import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectorMaos:
    def __init__(self, modo=False, max_maos=1, deteccao_confianca=0.5, rastreio_confianca=0.5, cor_pontos=(0, 255, 0), cor_linhas=(0, 0, 255)):

        """ 
        Função responsavel por inicializar a classe DetectorMaos
        :param modo: Modo de detecção (True para detecção contínua, False para detecção única)
        :param max_maos: Número máximo de mãos a serem detectadas
        :param deteccao_confianca: Percentual mínimo de confiança para considerar uma mão detectada. Se for menor que esse valor, a mão não será considerada detectada
        :param rastreio_confianca: Percentual da taxa de rastreio para considerar uma mão rastreada. Se for menor que esse valor, o rastreio dos pontos não será realizado
        :param cor_pontos: Cor dos pontos de referência das mãos (BGR)
        :param cor_linhas: Cor das linhas que conectam os pontos de referência das mãos
        """

        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_linhas = cor_linhas

        # Conexões oficiais dos pontos da mão (substitui HAND_CONNECTIONS)
        self.conexoes = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # Inicializa o detector da nova API
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
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

                    # Desenha os pontos de referência
                    for pt in pontos_px:
                        cv2.circle(imagem, pt, 5, self.cor_pontos, cv2.FILLED)

        return imagem        

    def dedos_levantados(self, lista_pontos):
        """
        Retorna uma lista de booleanos [polegar, indicador, medio, anelar, minimo]
        1 se o dedo estiver levantado, 0 se estiver dobrado.
        """
        if not lista_pontos:
            return []

        dedos = []
        pontas_ids = [8, 12, 16, 20]  # Pontas do Indicador, Médio, Anelar e Mínimo

        # 1. Polegar (Compara a posição do eixo X, dependendo do lado)
        if lista_pontos[4][1] < lista_pontos[3][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        # 2. Outros 4 dedos (Compara se a ponta Y está acima da articulação Y - 2 pontos abaixo)
        for id in pontas_ids:
            if lista_pontos[id][2] < lista_pontos[id - 2][2]:
                dedos.append(1)  # Levantado
            else:
                dedos.append(0)  # Dobrado

        return dedos

    def encontrarPontos(self, imagem, mao_num=0, desenho=True, cor=(255,0,255), raio=7, ponto_detectado=0):
        """
        Função responsável por encontrar os pontos de referência das mãos na imagem
        :param imagem: Imagem capturada
        :param mao_num: Número da mão a ser analisada (0 para a primeira mão, 1 para a segunda mão, etc.)
        :param desenho: desenhar o ponto encontrado
        :param cor: Cor dos pontos de referência das mãos (BGR)
        :param raio: Raio do circulo do ponto
        :param ponto_detectado: Ponto a ser detectado
        :return: Lista com pontos detectados
        """

        # Lista com os pontos detectados
        lista_pontos = []

        # Verifica se alguma mão foi detectada
        if self.resultados.hand_landmarks and len(self.resultados.hand_landmarks) > mao_num:
            # Obter os pontos da mão detectada, não de todas
            mao = self.resultados.hand_landmarks[mao_num]

            # Obter as informações de cada ponto da mão
            for id, ponto in enumerate(mao):

                # Obter as dimensões da imagem
                altura, largura, _ = imagem.shape
                centro_X, centro_Y = int(ponto.x * largura), int(ponto.y * altura)
                lista_pontos.append([id, centro_X, centro_Y])

                # Desenhar o ponto na imagem
                if desenho:
                    if id == ponto_detectado:
                        cv2.circle(
                            imagem, 
                            (centro_X, centro_Y),
                            raio, 
                            cor, 
                            cv2.FILLED
                            )

        return lista_pontos

def main():
    cap = cv2.VideoCapture(0)

    detector = DetectorMaos(cor_pontos=(0, 255, 0), cor_linhas=(0, 0, 255))

    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            break

        imagem = cv2.flip(imagem, 1)
        imagem = detector.encontrar_maos(imagem)
        
        # Pega as coordenadas dos 21 pontos da mão 0
        lista_pontos = detector.encontrarPontos(imagem, mao_num=0, desenho=False)

        if lista_pontos:
            # Identifica quais dedos estão levantados [Polegar, Indicador, Médio, Anelar, Mínimo]
            dedos = detector.dedos_levantados(lista_pontos)

            # --- LÓGICA DE RECONHECIMENTO DE GESTOS ---
            
            # Gesto 1: Aperta 'Joinha' / Polegar Levantado
            if dedos == [1, 0, 0, 0, 0]:
                y_ponta_polegar = lista_pontos[4][2]
                y_base_polegar = lista_pontos[2][2]

                # No OpenCV, valores de Y maiores ficam mais para BAIXO na tela
                if y_ponta_polegar > y_base_polegar:
                    cv2.putText(imagem, "DISLIKE (Polegar para baixo)", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(imagem, "LIKE (Polegar para cima)", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Gesto 2: Apontando (Apenas Indicador Levantado)
            elif dedos == [0, 1, 0, 0, 0]:
                cv2.putText(imagem, "Acao: APONTANDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Exemplo para TAA: mover um cursor virtual na tela

            # Gesto 3: Mão Aberta (Todos os dedos levantados)
            elif dedos == [1, 1, 1, 1, 1]:
                cv2.putText(imagem, "Acao: PARAR / MAO ABERTA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Gesto 4: Paz e Amor / Sinal de Vitória (Indicador e Médio)
            elif dedos == [0, 1, 1, 0, 0]:
                cv2.putText(imagem, "Acao: VITORIA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Gesto 5: Mão Fechada (Nenhum dedo levantado)
            elif dedos == [0, 0, 0, 0, 0]:
                cv2.putText(imagem, "Acao: AGARRAR / FECHADO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            elif dedos == [0, 1, 1, 1, 0]:
                cv2.putText(imagem, "Acao: W", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 60), 2)
                # Faça algo aqui: ex: mandar um comando de áudio, acionar um relé, etc.
            
            elif dedos == [1, 1, 0, 0, 0]:
                cv2.putText(imagem, "Acao: L", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 255, 200), 2)
                # Faça algo aqui: ex: mandar um comando de áudio, acionar um relé, etc.
                
            elif dedos == [1, 0, 0, 0, 1]:
                cv2.putText(imagem, "Acao: Hang Loose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 255, 200), 2)
                # Faça algo aqui: ex: mandar um comando de áudio, acionar um relé, etc.
            

        cv2.imshow("Reconhecimento de Gestos", imagem)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()