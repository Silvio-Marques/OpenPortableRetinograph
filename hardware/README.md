# Hardware

Esta pasta contém todos os arquivos relacionados ao desenvolvimento do hardware do **Open Portable Retinograph (OPR)**.

O projeto foi desenvolvido seguindo os princípios de **Hardware Aberto**, priorizando baixo custo, modularidade, facilidade de reprodução e utilização de componentes comerciais.

---

# Visão Geral

O hardware é composto por quatro subsistemas principais:

- Estrutura Mecânica
- Sistema Óptico
- Sistema Eletrônico
- Alimentação

Todos os componentes foram projetados para permitir manutenção, substituição e futuras melhorias sem necessidade de alterações estruturais significativas.

---

# Arquitetura Física

```
                 Smartphone
                (Interface Web)
                       │
                     Wi-Fi
                       │
               Raspberry Pi 4
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    Câmera M12      LEDs        Alimentação
        │
        ▼
 Sistema Óptico
(Espelhos + Lente 20D)
        │
        ▼
 Estrutura Mecânica
```

---

# Estrutura da Pasta

```
hardware/

├── CAD/
│   Modelos do projeto
│
├── STL/
│   Arquivos para impressão 3D
│
├── STEP/
│   Modelos CAD universais
│
├── BOM/
│   Lista de materiais
│
├── PCB/
│   Placas eletrônicas
│
├── Schematics/
│   Esquemas elétricos
│
├── Images/
│   Fotografias do protótipo
│
└── README.md
```

---

# Sistema Mecânico

O sistema mecânico foi desenvolvido em ambiente CAD e fabricado por impressão 3D.

Características:

- estrutura modular;
- baixo peso;
- fácil manutenção;
- ajuste de foco por deslocamento longitudinal da caixa óptica;
- apoio facial produzido em TPU.

---

# Sistema Óptico

O sistema óptico foi inspirado no **Open Indirect Ophthalmoscope (OIO)**.

É composto por:

- lente condensadora de 20 dioptrias;
- espelhos de primeira superfície;
- lente M12;
- caixa óptica;
- mecanismo de focalização.

O interior da caixa óptica recebeu pintura preta fosca para minimizar reflexões internas.

---

# Sistema Eletrônico

O sistema embarcado utiliza:

- Raspberry Pi 4;
- câmera M12;
- LEDs de alta potência;
- bateria recarregável;
- comunicação Wi-Fi.

A captura das imagens é realizada pela Raspberry Pi, enquanto toda a interação do usuário ocorre por meio de uma interface Web acessada por smartphone ou computador.

---

# Alimentação

O protótipo é alimentado por baterias recarregáveis, proporcionando aproximadamente **40 minutos de autonomia**.

A alimentação atende aos seguintes módulos:

- Raspberry Pi 4;
- câmera;
- LEDs;
- eletrônica auxiliar.

---

# Processo de Fabricação

1. Projeto CAD.
2. Impressão 3D das peças.
3. Acabamento das peças.
4. Montagem do sistema óptico.
5. Instalação da câmera.
6. Instalação da Raspberry Pi.
7. Integração elétrica.
8. Alinhamento óptico.
9. Ajuste do foco.
10. Testes de captura.

---

# Especificações

| Característica | Valor |
|----------------|------:|
| Tempo de desenvolvimento | 5 meses |
| Peso | ~800 g |
| Custo | US$ 491 |
| Peças impressas | 5 PLA + 2 TPU |
| Peça em MDF | 1 |
| Alimentação | Bateria recarregável |
| Autonomia | ~40 min |
| Comunicação | Wi-Fi |

---

# Lista de Materiais (BOM)

Os componentes utilizados encontram-se na pasta **BOM/**.

Principais componentes:

- Raspberry Pi 4
- Câmera M12
- Lente condensadora 20D
- Espelhos de primeira superfície
- LEDs de alta potência
- PLA
- TPU
- MDF
- Parafusos
- Cabos
- Bateria recarregável

---

# Estado Atual

| Componente | Status |
|------------|--------|
| Projeto CAD | ✅ |
| Impressão 3D | ✅ |
| Sistema Óptico | ✅ |
| Sistema Mecânico | ✅ |
| Raspberry Pi | ✅ |
| Sistema de Iluminação | ✅ |
| Ajuste de Foco | ✅ |
| Testes de Captura | ✅ |
| Versão Final | 🚧 |

---

# Trabalhos Futuros

- Redução do peso do equipamento;
- Miniaturização da estrutura;
- Otimização do sistema óptico;
- Desenvolvimento de uma PCB dedicada;
- Melhoria da autonomia da bateria;
- Adaptação para novos módulos de câmera;
- Preparação para fabricação em pequena escala.

---

# Licença

Este hardware é distribuído como projeto aberto.

Consulte o arquivo **LICENSE** para detalhes da licença utilizada.
