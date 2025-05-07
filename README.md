# Sana-Sprint local

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-orange.svg)

A user-friendly Gradio web interface for generating images using Efficient-Large-Model's [Sana Sprint](https://huggingface.co/Efficient-Large-Model) (0.6B & 1.6B) diffusers pipelines. This tool allows for easy interaction with these powerful few-step text-to-image models.

## Features

*   **Model Selection:** Choose between Sana Sprint 0.6B and 1.6B (1024px) models.
*   **Prompt Input:** Standard text prompt for image generation.
*   **Resolution Control:** Adjust image width and height (up to 1024px). *Note: Sana Sprint is optimized for 1024x1024.*
*   **Fixed Low Steps:** Generation is fixed at 2 inference steps, as Sana Sprint is designed for very few steps.
*   **Seed Control:** Set a specific seed for reproducible results or use -1 for a random seed.
*   **Batch Generation:** Generate multiple images from a single prompt.
*   **Memory Optimization:**
    *   **Precision Control:** Select `bf16`, `fp16`, or `fp32` (bf16 recommended for RTX 30+).
    *   **VAE Tiling:** Enable aggressive VAE tiling to reduce VRAM usage during VAE decoding.
    *   **Sequential CPU Offload:** Offload model parts to CPU to save VRAM (significantly slower, requires CUDA).
*   **Real-time Status & Logs:** Monitor model loading and image generation progress.
*   **Performance Stats:** View time taken for model loading and image generation.

## Prerequisites

*   Python 3.9+
*   NVIDIA GPU with CUDA support (for GPU acceleration and CPU offload). CPU-only mode is possible but very slow.

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/sana-sprint-gradio-generator.git
    cd sana-sprint-gradio-generator
    ```

2.  **Create and Activate a Virtual Environment:**
    *   **Using Conda (Recommended):**
        ```bash
        conda create -n sana_env python=3.10 -y
        conda activate sana_env
        ```
    *   **Using venv:**
        ```bash
        python3 -m venv sana_env
        source sana_env/bin/activate  # On Linux/macOS
        # sana_env\Scripts\activate.bat  # On Windows CMD
        # sana_env\Scripts\Activate.ps1  # On Windows PowerShell
        ```

3.  **Install PyTorch with CUDA Support:**
    Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to get the correct command for your system and CUDA version. For example, for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    Or for CUDA 12.1:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Hugging Face Login (Recommended):**
    To ensure you can download the models without issues, log in to Hugging Face CLI:
    ```bash
    huggingface-cli login
    ```
    You'll need a Hugging Face account and an access token with read permissions.

## Running the Application

1.  **(Optional but Recommended for Memory Issues) Set Environment Variable:**
    This can help with CUDA memory fragmentation.
    *   **Linux/macOS:**
        ```bash
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ```
    *   **Windows (CMD):**
        ```bash
        set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        ```
    *   **Windows (PowerShell):**
        ```bash
        $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        ```

2.  **Run the Gradio Script:**
    Ensure your virtual environment (`sana_env`) is activated.
    ```bash
    python run_sana_sprint.py
    ```

3.  **Access the UI:**
    Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Usage

1.  **Configure Model:**
    *   Select the desired Sana Sprint model (0.6B or 1.6B).
    *   Choose the precision (`bf16`, `fp16`, `fp32`).
    *   Enable/disable CPU Offload if needed.
    *   Click **"Load/Reload Model"**. Wait for the status message to confirm the model is ready. The "Generate Image(s)" button will become active.

2.  **Set Generation Parameters:**
    *   Enter your `Prompt`.
    *   Adjust `Width` and `Height`.
    *   Set the `Number of Images` to generate.
    *   Input a `Seed` (-1 for random).
    *   Toggle `VAE Tiling` if necessary for memory.

3.  **Generate:**
    *   Click **"Generate Image(s)"**.
    *   View the generated images in the gallery and check the status/logs for details.

## Troubleshooting

*   **`OutOfMemoryError` (OOM):**
    *   Try enabling "Sequential CPU Offload".
    *   Try enabling "Aggressive VAE Tiling".
    *   Use `fp16` precision instead of `bf16` or `fp32`.
    *   Use the 0.6B model instead of the 1.6B model.
    *   Reduce image resolution or the number of images generated simultaneously.
    *   Ensure no other applications are heavily using your GPU (check with `nvidia-smi`).
    *   Set the `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable.
    *   Restart the script/kernel to free all memory.
*   **Model Download Issues:**
    *   Ensure you have a stable internet connection.
    *   Verify you are logged in via `huggingface-cli login`.
    *   Some models might be gated; ensure your Hugging Face account has access.
*   **`TypeError: ... got an unexpected keyword argument 'tooltip'`:**
    *   Your Gradio version is too old. Upgrade it: `pip install --upgrade gradio`
    *   Alternatively, you can remove the `tooltip` argument from the `gr.Button("🎲", ...)` line in the script as a temporary fix.

## Acknowledgements

*   The [Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model) team for creating the Sana Sprint models.
*   [Hugging Face](https://huggingface.co/) for the `diffusers` library and model hosting.
*   The [Gradio](https://www.gradio.app/) team for the easy-to-use UI framework.
*   The [PyTorch](https://pytorch.org/) team.
