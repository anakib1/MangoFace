<h1>MangoFace</h1>

Framework for face clusterinzation with diverse model zoo.

<h2>Installation </h2>

```
pip intall -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

<h2> Example </h2>

- Create folder data with images in jpg format
- Run the following command:
```
python inference.py data -output_file o.csv
```
- Your results will be stored in `o.csv` file

<h2>Model zoo</h2>

**Clustering models**
- KMeans clustering
- DBScan clustering

**Embedding models**
- OpenAI CLIP
- VIT
- ResNet
- ResNeXT
