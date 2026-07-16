# Drawings

Esta pasta contém os desenhos de montagem utilizados na construção do retinógrafo portátil.

Os documentos apresentados servem como referência para a montagem mecânica, posicionamento dos componentes ópticos e integração do hardware embarcado.

## Arquivos

### Componentes.pdf

Vista explodida do retinógrafo contendo todos os componentes utilizados na montagem.

Os principais elementos identificados são:

| Identificação | Descrição |
|--------------|-----------|
| A | Apoio facial (TPU) |
| B | Caixa óptica |
| C | Carcaça principal |
| D | Suporte para smartphone |
| E | Adaptador para tela LCD (opcional) |
| F | Espelhos de primeira superfície |
| G | Suporte dos espelhos |
| H | Suporte da lente condensadora |
| I | Raspberry Pi 4 |
| J | Módulo de câmera M12 |
| K | Smartphone (interface Web) |
| L | Lente condensadora +20D |
| M | Placa eletrônica |
| N | Sistema de iluminação LED |

---

### Esquema.pdf

Desenho dimensional do protótipo.

Apresenta:

- dimensões externas;
- posicionamento dos subconjuntos;
- alinhamento óptico;
- localização da câmera;
- posição da lente condensadora;
- dimensões gerais do equipamento.

Dimensões aproximadas do protótipo:

- Comprimento: **26 cm**
- Altura: **17 cm**
- Largura: **11 cm**

---

# Sequência de montagem

A montagem recomendada do equipamento é realizada na seguinte ordem:

1. Impressão das peças em PLA e TPU.
2. Fabricação da base estrutural em MDF.
3. Montagem da caixa óptica.
4. Instalação dos espelhos de primeira superfície.
5. Instalação do suporte da lente condensadora.
6. Fixação da lente de +20 dioptrias.
7. Instalação da câmera M12.
8. Inserção da caixa óptica na carcaça principal.
9. Ajuste do mecanismo de focalização.
10. Instalação do sistema de iluminação LED.
11. Fixação da Raspberry Pi 4.
12. Conexão elétrica entre câmera, LEDs e Raspberry Pi.
13. Instalação da placa eletrônica.
14. Conexão da alimentação.
15. Testes de alinhamento óptico.
16. Ajuste fino do foco.
17. Conexão via Wi-Fi utilizando smartphone.
18. Testes finais de captura.

---

# Observações

- O apoio facial é produzido em TPU para proporcionar maior conforto e adaptação a diferentes formatos de rosto.
- O sistema foi projetado para utilização com smartphone como interface principal, reduzindo o custo do equipamento.
- A arquitetura também permite a instalação de uma tela LCD de 5" ou 3,5", embora esta configuração seja opcional.
- Todas as peças estruturais foram desenvolvidas para fabricação por impressão 3D, permitindo fácil reprodução e manutenção do equipamento.

---

# Documentação relacionada

Consulte também:

- `/hardware/CAD`
- `/hardware/STL`
- `/hardware/BOM`
- `/hardware/Schematics`
- `/hardware/Datasheets`
