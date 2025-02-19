#g++ -std=c++2a -O3 -ffast-math -march=native \
g++ -std=c++2a -O0 -g \
    -fconcepts \
    -I/home/bradley/miniconda3/envs/xtensor/include/ \
    -I/home/bradley/miniconda3/envs/xtensor/include/xtensor/ \
    -I/home/bradley/miniconda3/envs/xtensor/include/xtensor-blas/ \
    -Isrc \
    main.cpp \
    -L/home/bradley/miniconda3/envs/xtensor/lib \
    -lOpenImageIO \
    -lcblas \
    -o main

#-o main
