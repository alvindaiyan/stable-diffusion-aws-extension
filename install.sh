#!/bin/bash

INITIAL_SUPPORT_COMMIT_ROOT=68f336bd994bed5442ad95bad6b6ad5564a5409a
INITIAL_SUPPORT_COMMIT_CONTROLNET=efda6ddfd82ebafc6e1150fbb7e1f27163482a82
INITIAL_SUPPORT_COMMIT_DREAMBOOTH=c2a5617c587b812b5a408143ddfb18fc49234edf
INITIAL_SUPPORT_COMMIT_REMBG=3d9eedbbf0d585207f97d5b21e42f32c0042df70
INITIAL_SUPPORT_COMMIT_SAM=5df716be8445e0f358f6e8d4b65a87cc611bfe08
INITIAL_SUPPORT_COMMIT_TILEDVAE=f9f8073e64f4e682838f255215039ba7884553bf
INITIAL_SUPPORT_COMMIT_LORA_BLOCK_WEIGHT=3d226913c161032e578d9e122294c6490882f555
INITIAL_SUPPORT_COMMIT_ULTIMATE_UPSCALE=c99f382b31509b87b4d512e70e9caf08ae7a079f


# Clone stable-diffusion-webui
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# Go to stable-diffusion-webui directory
cd stable-diffusion-webui
# Reset to specific commit
git reset --hard ${INITIAL_SUPPORT_COMMIT_ROOT}

# Go to "extensions" directory
cd extensions

# Clone stable-diffusion-aws-extension
git clone https://github.com/awslabs/stable-diffusion-aws-extension.git

# Checkout aigc branch
cd stable-diffusion-aws-extension
cd ..

# Clone sd-webui-controlnet
git clone https://github.com/Mikubill/sd-webui-controlnet.git

# Go to sd-webui-controlnet directory and reset to specific commit
cd sd-webui-controlnet
git reset --hard ${INITIAL_SUPPORT_COMMIT_CONTROLNET}
cd ..

# Clone sd_dreambooth_extension
git clone https://github.com/d8ahazard/sd_dreambooth_extension.git

# Go to sd_dreambooth_extension directory and reset to specific commit
cd sd_dreambooth_extension
git reset --hard ${INITIAL_SUPPORT_COMMIT_DREAMBOOTH}
cd ..

# Clone stable-diffusion-webui-rembg
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg.git

# Go to stable-diffusion-webui-rembg directory and reset to specific commit
cd stable-diffusion-webui-rembg
git reset --hard ${INITIAL_SUPPORT_COMMIT_REMBG}
cd ..

# Clone sd-webui-segment-anything
git clone https://github.com/continue-revolution/sd-webui-segment-anything.git

# Go to sd-webui-segment-anything directory and reset to specific commit
cd sd-webui-segment-anything
git reset --hard ${INITIAL_SUPPORT_COMMIT_SAM}
cd ..

# Clone Tiled VAE
git clone https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111.git
cd multidiffusion-upscaler-for-automatic1111
git reset --hard ${INITIAL_SUPPORT_COMMIT_TILEDVAE}
cd ..

# Clone sd-webui-lora-block-weight
git clone https://github.com/hako-mikan/sd-webui-lora-block-weight.git
cd sd-webui-lora-block-weight
git reset --hard ${INITIAL_SUPPORT_COMMIT_LORA_BLOCK_WEIGHT}
cd ..


# Clone ultimate-upscale-for-automatic1111
git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git
cd ultimate-upscale-for-automatic1111
git reset --hard ${INITIAL_SUPPORT_COMMIT_ULTIMATE_UPSCALE}
cd ..
