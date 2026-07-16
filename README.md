# Open Portable Retinograph (OPR)

**Desenvolvimento de um retinógrafo portátil open source baseado em Raspberry Pi para aquisição de imagens retinianas e triagem automatizada de retinopatias.**

---

## Autores

**Desenvolvido por**

Silvio Marques (IFPE)

**Orientadores**

Tiago Lima (IFPE) 

Helio Bentzen (IFPE)

David Ribeiro (IFPE)

**Instituição**

IFPE CAMPUS PALMARES

Laboratório Maker campus Palmares

Brasil

---

# Sobre o Projeto

O **Open Portable Retinograph (OPR)** é um projeto de hardware aberto desenvolvido com o objetivo de disponibilizar uma plataforma portátil e de baixo custo para aquisição de imagens do fundo de olho.

O projeto integra manufatura aditiva, componentes ópticos comerciais e uma plataforma embarcada baseada em Raspberry Pi para permitir a captura de imagens retinianas utilizando hardware aberto, com foco em aplicações de pesquisa, ensino e futuras soluções de triagem automatizada de retinopatias.

A arquitetura foi concebida para reduzir o custo dos equipamentos convencionais, facilitar sua reprodução por outras instituições e servir como plataforma para integração futura de algoritmos de Inteligência Artificial embarcada.

---

# Licença

Este projeto é distribuído sob a licença **MIT License**.

Os arquivos CAD, documentação técnica, código-fonte e modelos computacionais podem ser utilizados, modificados e redistribuídos conforme os termos da licença.

---

# Isenção de Responsabilidade

Este equipamento destina-se exclusivamente a atividades de pesquisa, desenvolvimento tecnológico e ensino.

O protótipo **não possui certificação para uso clínico** e não deve ser utilizado para diagnóstico médico sem a devida aprovação pelos órgãos reguladores competentes.

---

# Software Ecosystem

O retinógrafo foi desenvolvido para operar de forma modular, permitindo integração com diferentes sistemas de Inteligência Artificial para análise das imagens adquiridas.

## OpenDR

Sistema para triagem automática de Retinopatia Diabética.

- Repositório: https://github.com/heliobentzen/openDR

Principais recursos:

- Captura automatizada
- Interface Web
- TensorFlow Lite
- Grad-CAM
- Relatórios
- Classificação automática

---

## Glaucoma AI

Sistema baseado em Deep Learning para classificação de glaucoma a partir de imagens retinianas.

- Demonstração:
https://huggingface.co/spaces/tiagopessoalim/glaucoma

Aplicação prevista:

- Segunda opinião diagnóstica
- Apoio ao rastreamento de glaucoma

---
# Versão

Versão atual

**v1.0.0 (Protótipo funcional)**

---

# Novidades da versão 1.0

- Desenvolvimento completo da arquitetura mecânica.
- Projeto CAD totalmente parametrizado.
- Sistema óptico baseado em lente condensadora de 20D.
- Integração de espelhos de primeira superfície.
- Estrutura produzida por impressão 3D.
- Apoio facial em TPU.
- Raspberry Pi 4 integrada ao sistema.
- Integração da câmera M12.
- Sistema de iluminação por LEDs de alta potência.
- Interface Web para operação remota.
- Comunicação Wi-Fi entre Raspberry Pi e smartphone.
- Captura de imagens retinianas.
- Protótipo funcional validado em bancada.

---

# Objetivos

- Desenvolver um retinógrafo portátil open source.
- Reduzir o custo de equipamentos para retinografia.
- Facilitar a reprodução da plataforma.
- Possibilitar futuras aplicações em Inteligência Artificial embarcada.
- Apoiar pesquisas em Oftalmologia e Engenharia Biomédica.

---

# Características

## Estrutura Mecânica

- Projeto CAD completo.
- Impressão 3D.
- Estrutura modular.
- Ajuste de foco por deslocamento da caixa óptica.
- Apoio facial em TPU.

---

## Sistema Óptico

- Lente condensadora de 20 dioptrias.
- Espelhos de primeira superfície.
- Lente M12.
- Pintura interna antirreflexo.
- Sistema alinhável.

---

## Hardware

- Raspberry Pi 4
- Módulo de câmera M12
- LEDs de alta potência
- Bateria recarregável
- Interface Wi-Fi

---

## Interface

- Captura remota
- Visualização em tempo real
- Armazenamento das imagens
- Controle via navegador
- Compatível com smartphones e computadores

---

# Especificações do Protótipo

| Característica | Valor |
|----------------|-------|
| Tempo de desenvolvimento | 5 meses |
| Peso | ~800 g |
| Custo | US\$ 491 |
| Estrutura | PLA + TPU + MDF |
| Comunicação | Wi-Fi |
| Interface | Web |
| Autonomia | ~40 minutos |
| Inicialização | ~2 minutos |
| Dispositivos simultâneos | 2 |

---

# Organização do Projeto

```
article/
Artigo científico

docs/
Documentação

hardware/
Arquivos CAD
STL
BOM

software/
Código Raspberry Pi
Interface Web

images/
Fotos
Resultados
CAD

tests/
Ensaios realizados

ai/
Modelos de IA
```

---

# Estado Atual

| Módulo | Status |
|---------|--------|
| Projeto CAD | ✅ |
| Sistema Óptico | ✅ |
| Impressão 3D | ✅ |
| Raspberry Pi | ✅ |
| Interface Web | ✅ |
| Captura de Imagens | ✅ |
| Testes Ópticos | ✅ |
| Inteligência Artificial |  |
| Validação Clínica |  |

---

# Trabalhos Futuros

- Otimização do sistema óptico.
- Desenvolvimento da versão miniaturizada.
- Integração de modelos TensorFlow Lite.
- Segmentação automática da retina.
- Classificação automática de retinopatias.
- Aplicativo móvel dedicado.
- Validação clínica.
- Publicação open hardware.

---

# Agradecimentos

Agradecemos às instituições e colaboradores envolvidos no desenvolvimento deste projeto, bem como à comunidade de Hardware Aberto e aos projetos Open Indirect Ophthalmoscope (OIO), Raspberry Pi Foundation e OpenCV, que serviram de inspiração para o desenvolvimento desta plataforma.

---

# Contato

Autor

smxj@discente.ifpe.edu.br

GitHub

https://github.com/usuario/OpenPortableRetinograph
