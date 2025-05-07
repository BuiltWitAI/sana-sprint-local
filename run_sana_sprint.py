# --- START OF FILE run_sana_sprint_gui_final.py ---

# BEFORE RUNNING THIS SCRIPT:
# 1. Ensure you have a Python environment (e.g., Conda or venv) set up with the required packages.
#    Let's assume this environment is named 'sana_env'.
#
# 2. Activate the environment:
#
#    If using Conda:
#    conda activate sana_env
#
#    If using venv (on Linux/macOS):
#    source path/to/sana_env/bin/activate
#
#    If using venv (on Windows CMD):
#    path\to\sana_env\Scripts\activate.bat
#
#    If using venv (on Windows PowerShell):
#    path\to\sana_env\Scripts\Activate.ps1
#
# 3. (Optional but Recommended for OOM issues) Set environment variable:
#    Linux/macOS: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#    Windows CMD:   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#    Windows PowerShell: $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#
# 4. (Optional) If you have an older Gradio version and see errors about 'tooltip',
#    you might need to upgrade it: pip install --upgrade gradio
#
# 5. Then, run this script from your activated environment:
#    python run_sana_sprint_gui_final.py
#
# The script starts below this line.
# ---------------------------------------------------------------------------

import torch
from diffusers import SanaSprintPipeline
import gradio as gr
import random
import os
import gc # For more explicit garbage collection
import time

# --- Configuration ---
MODEL_OPTIONS = {
    "Sana Sprint 0.6B (1024px)": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    "Sana Sprint 1.6B (1024px)": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
}
FIXED_NUM_STEPS = 2

# --- Global State ---
current_pipeline = None
current_model_id_loaded = None
current_dtype_loaded = None
current_device_loaded = "cpu"
pipeline_config_when_loaded = {}


# --- Helper Functions ---
def get_torch_dtype(dtype_str):
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    return torch.float32

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def check_bf16_support():
    if torch.cuda.is_available():
        try:
            return torch.cuda.is_bf16_supported()
        except AttributeError:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    return False

def get_random_seed_value():
    return -1

def actual_random_seed():
    return random.randint(0, 2**32 - 1)

# --- Core Logic ---

def load_model_action(model_name_key: str, dtype_str: str, enable_cpu_offload: bool, progress: gr.Progress = gr.Progress()):
    global current_pipeline, current_model_id_loaded, current_dtype_loaded, current_device_loaded, pipeline_config_when_loaded

    t_start_load = time.perf_counter()
    progress(0, desc="Initializing model load...")
    clear_gpu_cache()

    model_id = MODEL_OPTIONS[model_name_key]
    target_dtype = get_torch_dtype(dtype_str)
    initial_target_device = "cuda" if torch.cuda.is_available() and not enable_cpu_offload else "cpu"

    new_config = {"model_id": model_id, "dtype": target_dtype, "cpu_offload": enable_cpu_offload, "device": initial_target_device}
    if current_pipeline is not None and pipeline_config_when_loaded == new_config:
        t_end_load = time.perf_counter()
        status = f"Model '{model_id}' ({dtype_str}, CPU Offload: {enable_cpu_offload}) is already loaded. Load time: {t_end_load - t_start_load:.2f}s."
        progress(1, desc=status)
        return status, gr.update(interactive=True)

    status_message = ""
    try:
        progress(0.1, desc=f"Loading pipeline: {model_id} with {dtype_str}...")
        print(f"Loading pipeline for model: {model_id} with {target_dtype}...")

        if dtype_str == "bf16" and not check_bf16_support() and torch.cuda.is_available():
            warning_msg = "Warning: bfloat16 might not be optimally supported on your GPU. Consider fp16 or fp32."
            status_message += warning_msg + "\n"
            print(warning_msg)

        pipeline_args = {"torch_dtype": target_dtype}
        if os.environ.get("HF_TOKEN") is None and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
             print("Warning: Hugging Face token not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN for private/gated models.")

        if current_pipeline is not None:
            print(f"Unloading previous model: {current_model_id_loaded}")
            del current_pipeline
            current_pipeline = None
            clear_gpu_cache()
            progress(0.2, desc="Previous model unloaded. Clearing cache...")

        loaded_pipeline = SanaSprintPipeline.from_pretrained(model_id, **pipeline_args)
        status_message += f"Pipeline for {model_id} loaded from disk/hub.\n"
        progress(0.5, desc="Configuring device and offloading...")

        if enable_cpu_offload:
            if not torch.cuda.is_available():
                status_message += "CPU offload selected, but no CUDA device. Pipeline will run on CPU.\n"
                loaded_pipeline.to("cpu")
                current_device_loaded = "cpu"
            else:
                print("Enabling sequential CPU offload...")
                try:
                    loaded_pipeline.enable_sequential_cpu_offload(gpu_id=0)
                    status_message += "Sequential CPU offload enabled.\n"
                    current_device_loaded = loaded_pipeline.device # Device after offload
                except Exception as e_offload:
                    error_offload = f"Error enabling CPU offload: {e_offload}. Trying to use CUDA/CPU directly."
                    status_message += error_offload + "\n"
                    print(error_offload)
                    if torch.cuda.is_available(): loaded_pipeline.to("cuda"); current_device_loaded = "cuda"
                    else: loaded_pipeline.to("cpu"); current_device_loaded = "cpu"
        else:
            if initial_target_device == "cuda":
                print(f"Moving pipeline to {initial_target_device}...")
                try:
                    loaded_pipeline.to(initial_target_device)
                    current_device_loaded = initial_target_device
                    status_message += f"Pipeline moved to {current_device_loaded}.\n"
                except Exception as e_to_cuda:
                    error_cuda = f"Error moving pipeline to GPU: {e_to_cuda}. Falling back to CPU."
                    status_message += error_cuda + "\n"; print(error_cuda)
                    loaded_pipeline.to("cpu"); current_device_loaded = "cpu"
            else:
                loaded_pipeline.to("cpu"); current_device_loaded = "cpu"
                status_message += "Pipeline configured for CPU.\n"

        current_pipeline = loaded_pipeline
        current_model_id_loaded = model_id
        current_dtype_loaded = target_dtype
        pipeline_config_when_loaded = new_config.copy()
        pipeline_config_when_loaded["device"] = str(current_pipeline.device) # Store actual device

        t_end_load = time.perf_counter()
        status_message += f"Model ready. Load time: {t_end_load - t_start_load:.2f}s. Device: {current_pipeline.device}"
        progress(1, desc="Model ready.")
        return status_message, gr.update(interactive=True)

    except Exception as e:
        clear_gpu_cache()
        current_pipeline = None; pipeline_config_when_loaded = {}
        t_end_load = time.perf_counter()
        error_msg = f"Error loading model {model_id} (took {t_end_load - t_start_load:.2f}s): {e}\n"
        error_msg += "Check connection and Hugging Face login (huggingface-cli login)."
        print(error_msg)
        progress(1)
        # This will display the error in the Gradio UI's error modal
        # And the outputs (status_output, generate_btn) will not be updated if an error is raised.
        # generate_btn will remain in its previous state (likely disabled if this is the first load attempt).
        raise gr.Error(error_msg)


def generate_images_action(
    prompt: str,
    height: int,
    width: int,
    num_images: int,
    seed_val: int,
    enable_vae_tiling: bool,
    progress: gr.Progress = gr.Progress()
):
    global current_pipeline

    if not current_pipeline:
        raise gr.Error("Model not loaded. Please click 'Load/Reload Model' first.")
    if not prompt:
        raise gr.Error("Prompt cannot be empty.")
    if num_images < 1:
        raise gr.Error("Number of images must be at least 1.")

    t_start_gen = time.perf_counter()
    status_updates = f"Starting generation...\nModel: {current_model_id_loaded}\nDtype: {current_dtype_loaded}\nDevice: {current_pipeline.device}\n"
    status_updates += f"Prompt: '{prompt}'\nResolution: {width}x{height}\nNum Images: {num_images}\n"

    if seed_val == -1:
        used_seed = actual_random_seed()
        status_updates += f"Using random seed: {used_seed} (Input was -1)\n"
    else:
        used_seed = int(seed_val)
        status_updates += f"Using fixed seed: {used_seed}\n"
    
    generator = torch.Generator(device="cpu").manual_seed(used_seed)

    progress(0.1, desc="Configuring VAE tiling...")
    try:
        if enable_vae_tiling:
            current_pipeline.vae.enable_tiling(tile_sample_min_width=256, tile_sample_min_height=256, tile_sample_stride_height=192, tile_sample_stride_width=192)
            status_updates += "VAE tiling enabled for this generation.\n"
        else:
            if hasattr(current_pipeline.vae, 'disable_tiling'):
                current_pipeline.vae.disable_tiling()
                status_updates += "VAE tiling disabled for this generation.\n"
            else:
                status_updates += "VAE tiling cannot be disabled (method not found).\n"
    except AttributeError:
        status_updates += "Warning: VAE tiling attribute not found on pipeline.vae.\n"
    except Exception as e_vae:
        status_updates += f"Error configuring VAE tiling: {e_vae}\n"

    clear_gpu_cache()

    progress(0.3, desc=f"Generating {num_images} image(s) with {FIXED_NUM_STEPS} steps...")
    print(f"Generating with {FIXED_NUM_STEPS} steps, W:{width}, H:{height}, N:{num_images}")
    images_list = []
    try:
        output = current_pipeline(
            prompt=prompt,
            num_inference_steps=FIXED_NUM_STEPS,
            generator=generator,
            height=int(height),
            width=int(width),
            num_images_per_prompt=int(num_images)
        )
        images_list = output.images
        t_end_gen = time.perf_counter()
        gen_time = t_end_gen - t_start_gen
        status_updates += f"Generated {len(images_list)} image(s) successfully in {gen_time:.2f} seconds.\n"
        if num_images > 0 and len(images_list) > 0: # Avoid division by zero if no images generated
             status_updates += f"Time per image (approx): {gen_time / len(images_list):.2f} seconds.\n"
        progress(1, desc="Generation complete!")
        return images_list, status_updates
    except torch.cuda.OutOfMemoryError as oom_error:
        clear_gpu_cache(); t_end_gen = time.perf_counter()
        error_msg = f"CUDA OOM during generation (took {t_end_gen - t_start_gen:.2f}s): {oom_error}\n"
        error_msg += "Try: lower resolution/count, 'fp16' precision, enable CPU Offload, close other GPU apps.\n"
        status_updates += error_msg; print(error_msg); progress(1)
        return None, status_updates
    except Exception as e:
        clear_gpu_cache(); t_end_gen = time.perf_counter()
        error_msg = f"Error during generation (took {t_end_gen - t_start_gen:.2f}s): {e}\n"
        status_updates += error_msg; print(error_msg); progress(1)
        return None, status_updates


# --- Gradio UI ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.gr-button { color: white; border-color: #007bff; background: #007bff; }
.gr-button:hover { border-color: #0056b3; background: #0056b3; }
.status_box { font-family: 'monospace'; font-size: 0.9em; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; min-height:150px; overflow-y:auto; background-color:#f9f9f9;}
.attention { color: #d9534f; font-weight: bold; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Sana-Sprint Image Generator") as demo:
    gr.Markdown("# Sana-Sprint Image Generator")
    gr.Markdown(f"Generate images using Sana Sprint. Generation is fixed at **{FIXED_NUM_STEPS} steps**.")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Model Configuration")
            with gr.Group():
                model_select = gr.Dropdown(label="Select Model", choices=list(MODEL_OPTIONS.keys()), value=list(MODEL_OPTIONS.keys())[0])
                available_dtypes = ["bf16", "fp16", "fp32"]
                default_dtype = "bf16" if check_bf16_support() else "fp16"
                dtype_select = gr.Radio(available_dtypes, label="Precision (torch_dtype)",
                                        info=f"{'bf16 recommended for RTX 30+.' if check_bf16_support() else 'fp16 recommended.'}",
                                        value=default_dtype)
                enable_cpu_offload_checkbox = gr.Checkbox(label="Enable Sequential CPU Offload (Reduces VRAM, slower)",
                                                          value=False, interactive=torch.cuda.is_available(),
                                                          info="Requires CUDA. If disabled, model attempts GPU directly.")
                if not torch.cuda.is_available():
                    gr.Markdown("<p class='attention'>CUDA not available. CPU offload disabled. Model loads on CPU (slow).</p>")
            load_model_btn = gr.Button("Load/Reload Model", variant="secondary")

            gr.Markdown("### 2. Generation Settings")
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., a tiny astronaut...", lines=3,
                                      value="a tiny astronaut hatching from an egg on the moon, cinematic lighting, epic, 8k")
            with gr.Row():
                width_slider = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=1024)
                height_slider = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=1024)
            gr.Markdown("<p class='attention'>Note: Sana Sprint models are optimized for 1024x1024. Other resolutions might yield suboptimal results or fail.</p>")
            with gr.Row():
                num_images_slider = gr.Slider(label="Number of Images", minimum=1, maximum=10, step=1, value=1)
                seed_input = gr.Number(label="Seed (-1 for random)", value=get_random_seed_value(), precision=0)
                random_seed_btn = gr.Button("🎲", scale=0, elem_classes="gr-button-sm") # Tooltip removed for compatibility

            enable_vae_tiling_checkbox = gr.Checkbox(label="Enable Aggressive VAE Tiling (Reduces VAE VRAM)", value=True)
            generate_btn = gr.Button("Generate Image(s)", variant="primary", interactive=False)

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery", columns=[2], object_fit="contain", height="auto")
            status_output = gr.Textbox(label="Status & Logs", lines=10, interactive=False, elem_classes="status_box")

    # Event Handlers
    load_model_btn.click(
        fn=load_model_action,
        inputs=[model_select, dtype_select, enable_cpu_offload_checkbox],
        outputs=[status_output, generate_btn]
    )
    random_seed_btn.click(lambda: -1, outputs=seed_input)
    generate_btn.click(
        fn=generate_images_action,
        inputs=[prompt_input, height_slider, width_slider, num_images_slider, seed_input, enable_vae_tiling_checkbox],
        outputs=[output_gallery, status_output]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Important Notes:**\n"
        "- **Click 'Load/Reload Model'** whenever you change model, precision, or CPU offload settings.\n"
        f"- Generation uses **{FIXED_NUM_STEPS} steps**.\n"
        "- If OOM errors: try 'fp16' precision, enable 'CPU Offload', reduce image resolution/count, or use 0.6B model.\n"
        "- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (env var) can help with memory fragmentation."
    )
    gr.Markdown(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} (BF16 Support: {check_bf16_support()}) | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available. Operations will be on CPU.")

    current_pipeline = None
    pipeline_config_when_loaded = {}
    clear_gpu_cache()
    demo.launch(share=False)

# --- END OF FILE run_sana_sprint_gui_final.py ---
