#!bin/bash
# aim of the script is to
# 1. check whether conda version exits
# 2. if not, download it and install

conda_installation()
{
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then #  echo "Linux"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh";
     elif [[ "$OSTYPE" == "darwin"* ]]; then #   echo "Mac"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-MacOSX-x86_64.sh";
    fi

    wget $link -O anaconda.sh;
    bash anaconda.sh -b -p "$HOME/anaconda"
    source "$HOME/anaconda/etc/profile.d/conda.sh"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a
}

conda_installation