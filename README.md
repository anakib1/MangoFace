<h1>MangoFace</h1>

Framework for face clusterinzation with diverse model zoo.

<h2>Installation </h2>

```
pip intall -r requirements.txt
```

**Additional packages (required by some models):**

- For `CLIP` (Embedding):

```
pip install git+https://github.com/openai/CLIP.git
```

- For `DBScan` (Clustering)
```
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.2.* dask-cudf-cu12==24.2.* cuml-cu12==24.2.* \
    cugraph-cu12==24.2.* cuspatial-cu12==24.2.* cuproj-cu12==24.2.* \
    cuxfilter-cu12==24.2.* cucim-cu12==24.2.* pylibraft-cu12==24.2.* \
    raft-dask-cu12==24.2.*
```

- For `TSNECuda` (Vizualisation)
```
pip3 install tsnecuda==3.0.1+cu122 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
```

<h2> Example </h2>

- Create folder data with images in `.jpg` format
- Run the following command:
```
python inference.py data -output_file o.csv
```
- Your results will be stored in `o.csv` file

<h2>Model zoo</h2>

**Clustering models**
- KMeans 
- DBScan 
- HDBScan 

**Embedding models**
- OpenAI CLIP
- VIT
- ResNet
- ResNeXT
- VGG
- OpenFace

**Coming soon (after deadline :sad_face:)**
- GCN Clustering