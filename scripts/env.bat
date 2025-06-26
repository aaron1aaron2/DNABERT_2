@REM create and activate virtual python environment
conda create -n dna python=3.8
conda activate dna

@REM install required packages
pip install -r requirements.txt

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
@REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121