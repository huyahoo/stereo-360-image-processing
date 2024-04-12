# Generating Equirectangular / Cube Map Images

In time, this will become a collection of scripts to convert to and from equirectangular images and cube maps.

A more detailed explanation of how the script works can be found on my website, [www.paul-reed.co.uk/programming.htm](http://www.paul-reed.co.uk/programming.html)

## Python
### Example
- Equirectangular → Cube map
``` $bash
python3 CubemapFromEqui.py raw_data/PIC_1.jpg
```

- Cube map →　Equirectangular
``` $bash
python3 createEquiFromSquareFiles.py ./output/denoised/PIC_4_
```

## C++
### Usage
- Equirectangular → Cube map
``` $bash
./equi2cube <file_path>
```

- Cube map →　Equirectangular
``` $bash
./equi2cube <file_name>
```


### Example
- Equirectangular → Cube map
``` $bash
/usr/local/cuda-11.6/bin/nvcc -O3 equi2cube.cu `pkg-config opencv4 --cflags --libs` -o equi2cube
./equi2cube ./raw_data/PIC_4.jpg
```

- Cube map →　Equirectangular
``` $bash
/usr/local/cuda-11.6/bin/nvcc -O3 cube2equi.cu `pkg-config opencv4 --cflags --libs` -o cube2equi
./cube2equi PIC_4
```

### Denoise
- Denoise utilizing CUDA
``` $bash
/usr/local/cuda-11.6/bin/nvcc -O3 denoise-cuda.cu `pkg-config opencv4 --cflags --libs` -o denoise-cuda
./denoise-cuda ./output/cube/PIC_4_negx.jpg 5 1
./denoise-cuda ./output/cube/PIC_4_negy.jpg 5 1
./denoise-cuda ./output/cube/PIC_4_negz.jpg 5 1
./denoise-cuda ./output/cube/PIC_4_posx.jpg 5 1
./denoise-cuda ./output/cube/PIC_4_posy.jpg 5 1
./denoise-cuda ./output/cube/PIC_4_posz.jpg 5 1


```

- Denoise utilizing OPEN OMP
``` $bash
g++ denoise-omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ denoise-omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o denoise-omp
./denoise-omp ./output/cube/PIC_4_negx.jpg 3 3
./denoise-omp ./output/cube/PIC_4_negy.jpg 3 3
./denoise-omp ./output/cube/PIC_4_negz.jpg 3 3
./denoise-omp ./output/cube/PIC_4_posx.jpg 3 3
./denoise-omp ./output/cube/PIC_4_posy.jpg 3 3
./denoise-omp ./output/cube/PIC_4_posz.jpg 3 3
```

