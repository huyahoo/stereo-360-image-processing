# Generating Equirectangular / Cube Map Images

In time, this will become a collection of scripts to convert to and from equirectangular images and cube maps.

A more detailed explanation of how the script works can be found on my website, [www.paul-reed.co.uk/programming.htm](http://www.paul-reed.co.uk/programming.html)

## Python
### Example
- Equirectangular → Cube map
``` $bash
python3 createEquiFromSquareFiles.py raw_data/PIC_1.jpg
```

- Cube map →　Equirectangular
``` $bash
python3 CubemapFromEqui.py raw_data/PIC_1.jpg
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

