# Software

O software do **Open Portable Retinograph** é responsável pelo controle do hardware, aquisição das imagens retinianas, comunicação com o usuário e execução dos algoritmos de processamento e Inteligência Artificial.

A arquitetura foi desenvolvida de forma modular, permitindo que cada componente evolua independentemente.

---

## Estrutura

```
software/
├── Interface Web/
│   └── openDR/
│
├── RaspberryPi/
│
├── glaucoma/
│
└── README.md
```

---

## Diretórios

### Interface Web/openDR/

Contém a interface gráfica utilizada para operar o retinógrafo.

Principais funcionalidades:

- Visualização em tempo real da câmera;
- Captura de imagens;
- Controle remoto via navegador;
- Comunicação Wi-Fi com a Raspberry Pi;
- Armazenamento das imagens;
- Interface responsiva para smartphones e computadores;
- Integração com o pipeline de processamento.

Tecnologias utilizadas:

- Python
- Flask
- HTML
- CSS
- JavaScript
- OpenCV

---

### RaspberryPi/

Contém todos os programas executados diretamente na Raspberry Pi.

Responsabilidades:

- Inicialização do sistema;
- Controle da câmera M12;
- Controle da iluminação LED;
- Comunicação GPIO;
- Gerenciamento do armazenamento;
- Configuração da rede Wi-Fi;
- Comunicação com a Interface Web.

---

### glaucoma/

Contém os algoritmos de Inteligência Artificial utilizados para processamento das imagens retinianas.

Inclui:

- modelos treinados;
- pré-processamento;
- segmentação;
- classificação;
- geração de mapas Grad-CAM;
- inferência local.

Os modelos podem ser executados localmente na Raspberry Pi ou remotamente por meio da plataforma Hugging Face Spaces.

---

## Fluxo do software

```
Raspberry Pi
      │
      ▼
Interface Web
      │
      ▼
Aquisição da imagem
      │
      ▼
Pré-processamento
      │
      ▼
Inteligência Artificial
      │
      ▼
Resultado
```

---

## Dependências

- Python 3.11+
- Flask
- OpenCV
- NumPy
- Picamera2
- libcamera
- pigpio
- Requests
- Torch
- Torchvision
- TensorFlow Lite

---

## Execução

1. Inicializar a Raspberry Pi;
2. Executar o servidor Flask;
3. Conectar um smartphone ou computador via Wi-Fi;
4. Acessar a interface Web;
5. Capturar a imagem retiniana;
6. Executar o processamento e a inferência.

---

## Desenvolvimento

Cada módulo pode ser atualizado independentemente.

A organização em diretórios facilita:

- manutenção;
- testes;
- integração contínua;
- reutilização dos componentes;
- desenvolvimento colaborativo.

---

## Projetos relacionados

- OpenDR
- Hugging Face Space (Glaucoma)
- Raspberry Pi OS
- OpenCV
