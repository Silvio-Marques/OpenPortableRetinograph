import base64
import io
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import cv2
import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ======================================================================
# Configuração Centralizada
# ======================================================================
@dataclass
class Config:
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG_PATH: Path = Path("./models/convnext_tiny.json")
    WEIGHTS_PATH: Path = Path("./models/convnext_tiny.pt")
    
    RG_THRESHOLD: float = 0.508
    FEATURE_THRESHOLDS: Tuple[float, ...] = (
        0.508, 0.37, 0.614, 0.676, 0.697, 
        0.75, 0.754, 0.171, 0.531, 0.814
    )
    
    FEATURE_LABELS_PT: Dict[str, str] = None
    EXAMPLE_FILES: list = None

    def __post_init__(self):
        self.FEATURE_LABELS_PT = {
            "appearance neuroretinal rim superiorly": "Anel neuro‑retiniano superior",
            "appearance neuroretinal rim inferiorly": "Anel neuro‑retiniano inferior",
            "retinal nerve fiber layer defect superiorly": "Defeito CFN superior",
            "retinal nerve fiber layer defect inferiorly": "Defeito CFN inferior",
            "baring of the circumlinear vessel superiorly": "Exposição vaso circunlinear sup.",
            "baring of the circumlinear vessel inferiorly": "Exposição vaso circunlinear inf.",
            "nasalization of the vessel trunk": "Nasalização do tronco vascular",
            "disc hemorrhages": "Hemorragias de disco",
            "laminar dots": "Pontos laminares",
            "large cup": "Escavação aumentada",
        }
        self.EXAMPLE_FILES = [f"examples/{i}.jpg" for i in range(8)]

cfg = Config()
CSS_PATH = Path("./style.css")

# ======================================================================
# Pré‑processamento
# ======================================================================
def crop_black_borders(img: np.ndarray, threshold: int = 7) -> np.ndarray:
    """Corta as bordas pretas de uma imagem retinográfica."""
    if img.ndim == 2:
        mask = img > threshold
        return img[np.ix_(mask.any(1), mask.any(0))]
    
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > threshold
        coords = np.ix_(mask.any(1), mask.any(0))
        if img[coords].size == 0:
            return img
        return np.stack([img[..., c][coords] for c in range(3)], axis=-1)
    
    return img

TEST_TRANSFORM = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ======================================================================
# Modelo
# ======================================================================
class GlaucomaModel:
    """Classe responsável pelo carregamento e inferência do modelo."""
    def __init__(self, cfg_path: Path, weights_path: Path):
        try:
            with open(cfg_path) as f:
                self.model_cfg = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Arquivo de configuração não encontrado: {cfg_path}. Usando valores padrão.")
            self.model_cfg = {"model_name": "convnext_tiny", "IMG_SIZE": 224}
            
        self.model_cfg["checkpoint"] = str(weights_path)
        self.model = self._build_model()
        self.model.to(cfg.DEVICE)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        model = timm.create_model(self.model_cfg["model_name"], pretrained=False, num_classes=11)
        model.head.fc = nn.Sequential(model.head.fc, nn.Sigmoid())
        
        try:
            state_dict = torch.load(self.model_cfg["checkpoint"], map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, TypeError):
            state_dict = torch.load(self.model_cfg["checkpoint"], map_location="cpu", weights_only=False)
        except FileNotFoundError:
            logging.error(f"Pesos do modelo não encontrados em {self.model_cfg['checkpoint']}")
            return model # Retorna modelo não inicializado para evitar crash imediato se não existir
            
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
            model.head.fc.weight = nn.Parameter(state_dict["head.fc.0.weight"])
            model.head.fc.bias = nn.Parameter(state_dict["head.fc.0.bias"])
        return model

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        return self.model(input_tensor.to(cfg.DEVICE)).squeeze(0).cpu().numpy()

# Inicialização global do modelo
model = GlaucomaModel(cfg.CFG_PATH, cfg.WEIGHTS_PATH)

# ======================================================================
# Grad‑CAM (Processamento e Renderização)
# ======================================================================
def _pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def generate_gradcam_overlay(np_img: np.ndarray, input_tensor: torch.Tensor) -> Optional[Image.Image]:
    """Calcula o mapa de ativação Grad-CAM e retorna a imagem sobreposta."""
    target_layer = model.model.stages[-1]
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.model.eval()
        with torch.enable_grad():
            output = model.model(input_tensor.to(cfg.DEVICE))
            model.model.zero_grad()
            output[0, 0].backward()

        acts = activations["value"].detach()
        grads = gradients["value"].detach()

        if acts.shape[1] <= acts.shape[-1]:
            weights = grads.mean(dim=(1, 2))[0]
            cam = (acts[0] * weights).sum(dim=-1)
        else:
            weights = grads.mean(dim=(2, 3))[0]
            cam = (acts[0] * weights[:, None, None]).sum(0)

        cam_np = cam.cpu().float().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_np /= cam_np.max() + 1e-8

        h, w = np_img.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = (0.45 * np_img + 0.55 * heatmap).clip(0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
        
    except Exception as e:
        logging.error(f"Erro ao computar Grad-CAM: {e}")
        return None
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

def _render_gradcam_html(img_orig: Image.Image, img_overlay: Optional[Image.Image]) -> str:
    """Gera o HTML do slider comparativo do Grad-CAM."""
    if img_overlay is None:
        return '<p class="gradcam-error">Grad‑CAM indisponível para esta imagem.</p>'
        
    b64_orig = _pil_to_base64(img_orig)
    b64_over = _pil_to_base64(img_overlay)
    uid = f"gc{abs(hash(b64_over[:20])):x}"

    return f"""
    <div class="gradcam-card">
        <p class="gradcam-title">Mapa de atenção (Grad-CAM)</p>
        <div class="gradcam-compare" style="--pos: 0%;" id="{uid}">
            <img class="gradcam-base" src="data:image/png;base64,{b64_orig}" alt="Original" />
            <div class="gradcam-layer" style="background-image:url('data:image/png;base64,{b64_over}');"></div>
            <input type="range" min="0" max="100" value="0" class="gradcam-hitarea"
                aria-label="Controle do comparador Grad-CAM"
                oninput="this.parentElement.style.setProperty('--pos', this.value+'%')" />
            <div class="gradcam-handle"><span>||</span></div>
        </div>
        <p class="gradcam-hint">Clique e arraste diretamente sobre a imagem para revelar o mapa de calor</p>
    </div>"""

# ======================================================================
# Lógica da Aplicação (Processamento e UI Render)
# ======================================================================
def load_image(image_input: Union[str, Image.Image, dict, None]) -> Image.Image:
    if image_input is None:
        raise ValueError("Nenhuma imagem fornecida.")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, dict) and image_input.get("path"):
        return Image.open(image_input["path"]).convert("RGB")
    raise ValueError("Formato de arquivo não suportado.")

def predict(image_input: Union[str, Image.Image, dict, None]) -> Tuple[str, str, str, gr.update]:
    """Fluxo principal: pré-processa, prediz e gera os componentes HTML de saída."""
    if image_input is None:
        placeholder = "<div class='result-card placeholder'>Por favor, envie uma imagem de fundo de olho.</div>"
        return placeholder, "", "", gr.update(visible=True)

    try:
        pil_image = load_image(image_input)
    except ValueError:
        error_card = "<div class='result-card error'>Formato de arquivo não suportado ou vazio.</div>"
        return error_card, "", "", gr.update(visible=True)

    # Pre-processamento
    np_img = np.array(pil_image)
    np_img = crop_black_borders(np_img)
    img_size = model.model_cfg.get("IMG_SIZE", 224)
    np_img_resized = cv2.resize(np_img, (img_size, img_size))

    # Inferência
    tensor = TEST_TRANSFORM(image=np_img_resized)["image"].unsqueeze(0)
    avg_output = model.predict(tensor)

    # Análise de resultados
    rg_prob = float(avg_output[0])
    just_probs = avg_output[1:]
    rg_positive = rg_prob > cfg.RG_THRESHOLD

    features = {
        name: bool(just_probs[i] > cfg.FEATURE_THRESHOLDS[i])
        for i, name in enumerate(cfg.FEATURE_LABELS_PT)
    }

    result = {
        "referable_glaucoma_likelihood": rg_prob,
        "threshold": cfg.RG_THRESHOLD,
        "positive": rg_positive,
        "features": features,
    }

    # Renderização
    main_html = _render_main_result(result)
    lesions_html = _render_features(result)
    
    # Grad-CAM
    img_orig = Image.fromarray(np_img_resized.astype(np.uint8))
    img_overlay = generate_gradcam_overlay(np_img_resized, tensor)
    gradcam_html = _render_gradcam_html(img_orig, img_overlay)

    # Esconde o componente de input (gr.update) e mostra o gradcam no lugar
    return main_html, lesions_html, gradcam_html, gr.update(visible=False)

def reset_interface() -> Tuple[str, str, str, gr.update]:
    """Reseta a interface para uma nova análise."""
    return "", "", "", gr.update(visible=True, value=None)

def _render_main_result(result: dict) -> str:
    prob = result["referable_glaucoma_likelihood"]
    limiar = result["threshold"]
    margin = prob - limiar
    positive = result["positive"]

    if positive:
        verdict = "Suspeita positiva para glaucoma referível"
        level = "Alto" if margin >= 0.08 else "Intermediário"
        color = "#b91c1c"
    else:
        verdict = "Baixa suspeita"
        level = "Baixo"
        color = "#166534"

    return f"""
    <div class="result-card">
        <div class="verdict" style="color: {color}; font-weight: 700; font-size: 1.25rem;">
            {verdict}
        </div>
        <div class="details">
            <div class="detail-row"><span class="label">Probabilidade:</span> <span class="value">{prob:.4f}</span></div>
            <div class="detail-row"><span class="label">Limiar:</span> <span class="value">{limiar:.4f}</span></div>
            <div class="detail-row"><span class="label">Nível de alerta:</span> <span class="value">{level}</span></div>
        </div>
        <div class="disclaimer">
            Apoio à triagem. Interpretação clínica sempre necessária.
        </div>
    </div>"""

def _render_features(result: dict) -> str:
    rows = []
    for eng, pt in cfg.FEATURE_LABELS_PT.items():
        present = result["features"].get(eng, False)
        status = "Presente" if present else "Ausente"
        row_class = "feature-present" if present else "feature-absent"
        rows.append(
            f'<tr class="{row_class}"><td>{pt}</td>'
            f'<td><span class="status-badge {"active" if present else "inactive"}">{status}</span></td></tr>'
        )
    return f"""
    <div class="lesions-card">
        <h3>Sinais morfológicos avaliados</h3>
        <table>
            <thead><tr><th>Sinal</th><th>Decisão</th></tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>"""

# ======================================================================
# Instância da Interface Gradio
# ======================================================================
with gr.Blocks(css=CSS_PATH.read_text(encoding="utf-8"), theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero">
        <h1>Triagem Inteligente de Glaucoma</h1>
        <p>Análise automatizada de retinografia com mapa de atenção</p>
    </div>
    <div class="warning-box">
        ⚠️ <strong>Aviso clínico:</strong> Esta demonstração é voltada para pesquisa e não substitui a avaliação médica.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Image(
                type="filepath",
                sources=["upload"],
                image_mode="RGB",
                label="Imagem de fundo de olho",
            )
            gradcam_out = gr.HTML()

            with gr.Row():
                gr.Examples(
                    examples=cfg.EXAMPLE_FILES,
                    inputs=inp,
                    label="Exemplos rápidos",
                    examples_per_page=4,
                )

        with gr.Column(scale=2):
            with gr.Row():
                run_btn = gr.Button("🔍 Analisar", variant="primary")
                reset_btn = gr.Button("↺ Nova análise", variant="secondary")
            
            main_txt = gr.HTML()
            lesions_tbl = gr.HTML()
            
            gr.HTML("""
            <div class="instruction-box">
                <strong>Como interpretar:</strong>
                <ol>
                    <li>Probabilidade acima do limiar → suspeita positiva.</li>
                    <li>A tabela lista sinais individuais detectados.</li>
                    <li>Utilize sempre com avaliação oftalmológica completa.</li>
                </ol>
            </div>
            """)

    # Ligação de Eventos
    run_btn.click(
        fn=predict,
        inputs=inp,
        outputs=[main_txt, lesions_tbl, gradcam_out, inp],
    )
    
    inp.change(
        fn=predict,
        inputs=inp,
        outputs=[main_txt, lesions_tbl, gradcam_out, inp],
    )
    
    reset_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[main_txt, lesions_tbl, gradcam_out, inp],
    )

if __name__ == "__main__":
    demo.queue().launch()