#!bin/bash
# aim of the script is to
# 1. check whether conda version exits
# 2. if not, download it and install

conda_installation()
{
    download_dir="temp" # check which OS to define download link and name of installation file
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then #  echo "Linux"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh";
     elif [[ "$OSTYPE" == "darwin"* ]]; then #   echo "Mac"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-MacOSX-x86_64.sh";
    fi

    wget $link -O anaconda.sh;
    bash anaconda.sh -b -p "$HOME/anaconda"
#    source "$HOME/anaconda/etc/profile.d/conda.sh"
#    hash -r
#    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a

#    if [ -d "$download_dir" ]; then # check whether directory temp (for download) exists: -d
#       if [ "$(ls -A "$download_dir")" ]; then # Check whether existing directory is empty
#         empty=false
#         for file in "$download_dir"/*; # check whether installation file is in $download_dir
#         do
#           if [[ ${file: -3} == ".sh" ]]; then
#             installation_file=$file
#           fi
#         done;
#       else
#         empty=true
#       fi
#    else
#       mkdir "temp"
#       empty=true
#    fi

#   if [ $empty != false ]; then # if $download_dir is empty, download file
#     wget $link -P $download_dir -O anaconda.sh
#   fi

#   printf 'yes\nqyes\n\n' | bash ./"$download_dir/$installation_file" # install file
#   ( echo -ne "yes\n" ; echo -ne "\n"; echo -ne "\n" ) | bash ./"$download_dir/$installation_file" # install file
#   echo -ne '\n' | <yourfinecommandhere>
}
#
#if [[ "$(conda --version)" == "conda "* ]];
#then
# echo "Success."
#else
#  echo "Call installation function."
#  conda_installation()
#fi

conda_installation