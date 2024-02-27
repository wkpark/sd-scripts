import gradio as gr
from .common_gui import (
    get_any_file_path,
    get_folder_path,
    set_pretrained_model_name_or_path_input,
)

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄


class SourceModel:
    def __init__(
        self,
        save_model_as_choices=[
            'same as source model',
            'ckpt',
            'diffusers',
            'diffusers_safetensors',
            'safetensors',
        ],
        headless=False,
    ):
        self.headless = headless
        self.save_model_as_choices = save_model_as_choices

        default_models = [
            'stabilityai/stable-diffusion-xl-base-1.0',
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            'stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned',
            'stabilityai/stable-diffusion-2-1-base',
            'stabilityai/stable-diffusion-2-base',
            'stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned',
            'stabilityai/stable-diffusion-2-1',
            'stabilityai/stable-diffusion-2',
            'runwayml/stable-diffusion-v1-5',
            'CompVis/stable-diffusion-v1-4',
        ]
        try:
            from modules import sd_models, shared
            model_checkpoint_tiles = sd_models.checkpoint_tiles()
        except Exception:
            model_checkpoint_tiles = []
            pass

        with gr.Column():
            # Define the input elements
            with gr.Row():
                self.model_list = gr.Textbox(visible=False, value="")
                self.pretrained_model_name_or_path = gr.Dropdown(
                    label='Pretrained model name or path',
                    choices=default_models + model_checkpoint_tiles,
                    value='runwayml/stable-diffusion-v1-5',
                    allow_custom_value=True,
                    visible=True,
                )
                self.pretrained_model_name_or_path_file = gr.Button(
                    document_symbol,
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                    visible=True,
                )
                self.pretrained_model_name_or_path_file.click(
                    get_any_file_path,
                    inputs=self.pretrained_model_name_or_path,
                    outputs=self.pretrained_model_name_or_path,
                    show_progress=False,
                )
                self.pretrained_model_name_or_path_folder = gr.Button(
                    folder_symbol,
                    elem_id='open_folder_small',
                    elem_classes=['tool'],
                    visible=True,
                )
                self.pretrained_model_name_or_path_folder.click(
                    get_folder_path,
                    inputs=self.pretrained_model_name_or_path,
                    outputs=self.pretrained_model_name_or_path,
                    show_progress=False,
                )
                self.save_model_as = gr.Dropdown(
                    label='Save trained model as',
                    choices=save_model_as_choices,
                    value='safetensors',
                )

            with gr.Row():
                self.v2 = gr.Checkbox(label='v2', value=False, visible=False)
                self.v_parameterization = gr.Checkbox(
                    label='v_parameterization', value=False, visible=False
                )
                self.sdxl_checkbox = gr.Checkbox(
                    label='SDXL Model', value=False, visible=False
                )
