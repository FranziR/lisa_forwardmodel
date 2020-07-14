#!bin/bash
# aim of the script is to
# 1. check whether conda version exits
# 2. if not, download it and install

conda_installation()
{
    download_dir="temp" # check which OS to define download link and name of installation file
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then #  echo "Linux"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-ppc64le.sh"
        installation_file="Anaconda3-2020.02-Linux-ppc64le.sh"
     elif [[ "$OSTYPE" == "darwin"* ]]; then #   echo "Mac"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-MacOSX-x86_64.sh"
        installation_file="Anaconda3-2020.02-MacOSX-x86_64.sh"
     elif [[ "$OSTYPE" == "win32" ]]; then #   echo "Windows"
        link="https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86.exe"
        installation_file="Anaconda3-2020.02-Windows-x86.exe"
    fi

    if [ -d "$download_dir" ]; then # check whether directory temp (for download) exists: -d
       if [ "$(ls -A "$download_dir")" ]; then # Check whether existing directory is empty
         empty=false
         for file in "$download_dir"/*; # check whether installation file is in $download_dir
         do
           if [[ ${file: -3} == ".sh" ]] || [[ ${file: -4} == ".exe" ]]; then
             installation_file=$file
           fi
         done;
       else
         empty=true
       fi
    else
       mkdir "temp"
       empty=true
    fi

   if [ $empty != false ]; then # if $download_dir is empty, download file
     wget $link -P $download_dir
   fi

#   printf 'yes\nqyes\n\n' | bash ./"$download_dir/$installation_file" # install file
#   ( echo -ne "yes\n" ; echo -ne "\n"; echo -ne "\n" ) | bash ./"$download_dir/$installation_file" # install file
   bash ./"$download_dir/$installation_file" -b >/dev/null # install file
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