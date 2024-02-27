import html
import json
import re
import os
import gradio as gr

from copy import copy
from kohya_gui.dreambooth_gui import dreambooth_tab
from kohya_gui.finetune_gui import finetune_tab
from kohya_gui.textual_inversion_gui import ti_tab
from kohya_gui.lora_gui import lora_tab
from kohya_gui.class_lora_tab import LoRATools
from kohya_gui.utilities import utilities_tab

from library.utils import setup_logging

from modules import script_callbacks, shared

# Set up logging
log = setup_logging()


def add_kohya_tab():
    headless = False

    with gr.Blocks(title=f"Kohya GUI") as kohya_gui:
        with gr.Tab("Dreambooth"):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = dreambooth_tab(headless=headless)
        with gr.Tab("LoRA"):
            lora_tab(headless=headless)
        with gr.Tab("Textual Inversion"):
            ti_tab(headless=headless)
        with gr.Tab("Finetuning"):
            finetune_tab(headless=headless)
        with gr.Tab("Utilities"):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            with gr.Tab("LoRA"):
                _ = LoRATools(headless=headless)
#        with gr.Tab("About"):
#            gr.Markdown(f"kohya_ss GUI release {release}")
#            with gr.Tab("README"):
#                gr.Markdown(README)

    return [(kohya_gui, "Kohya GUI", "koyha_train")]

#def on_ui_settings():
#    shared.opts.add_option("pnginfo_diff_default_neg_prompt", shared.OptionInfo(DEFAULT_NEGATIVE, "Default Negative prompt", section=("pnginfo_diff", "PNGInfo Diff")))

#script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(add_kohya_tab)
