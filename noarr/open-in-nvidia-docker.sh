# Prerequisite:
# You need to have "nvidia/cuda:10.2-devel-ubuntu18.04" image pulled.
#
# Simply run this script and go to the /noarr folder.
# Now you can call "make" and it will compile.

docker run --rm -it -v $(pwd):/noarr nvidia/cuda:10.2-devel-ubuntu18.04 bash
