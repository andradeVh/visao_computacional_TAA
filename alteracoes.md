# Registro de Alterações - Eduardo

## Refatoração de Estrutura e Prevenção de Erros

* **Inicialização Segura de Atributos:** 
  * O atributo `self.resultados` foi declarado explicitamente como `None` dentro do método `__init__` da classe `DetectorMaos`. Isso previne interrupções do tipo `AttributeError` no método `encontrarPontos` caso a leitura da webcam falhe no primeiro frame ou tente buscar os pontos antes que qualquer mão seja identificada.
  * O método `encontrarPontos` recebeu uma validação extra de objeto (`if self.resultados and ...`) antes de tentar extrair as coordenadas (landmarks).
  
* **Validação do Modelo de Inteligência Artificial Local:**
  * Foram incluídas as importações das bibliotecas nativas `os` e `sys`.
  * Antes da inicialização das configurações do MediaPipe (`HandLandmarkerOptions`), o sistema agora verifica de forma proativa se o arquivo `hand_landmarker.task` existe no diretório raiz do projeto.
  * Caso o arquivo esteja ausente, o programa emite um alerta legível no terminal orientando a leitura do `README.md` e interrompe a execução graciosamente usando `sys.exit(1)`, evitando o "crash" genérico e verboso do MediaPipe C++.

* **Gerenciamento de Recursos e Fluxo da Câmera (Saída Segura):**
  * O laço infinito no `main()` passou de `while True:` para `while cap.isOpened():`, garantindo que o programa só tente processar frames se houver uma via de comunicação ativa com a webcam.
  * Foi introduzida uma quebra de segurança `if not sucesso: break` após o `cap.read()`.
  * Inserida a lógica de parada segura na linha `if cv2.waitKey(1) & 0xFF == 27: break`, permitindo ao usuário encerrar a captura de forma controlada pressionando a tecla `ESC`.
  * Adicionados os métodos `cap.release()` e `cv2.destroyAllWindows()` fora do laço para garantir que a memória do sistema, a webcam e as threads do OpenCV sejam liberadas corretamente pelo Sistema Operacional.