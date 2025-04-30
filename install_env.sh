#In case conda is not instlled
#
#curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh; chmod +x; miniconda.sh; sh miniconda.sh -b -p /content/miniconda
#export PATH="/content/miniconda/bin:$PATH"

conda create -n finetune python=3.10

conda activate finetune

pip install -r requirements.txt
