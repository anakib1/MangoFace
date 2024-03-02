from argparse import ArgumentParser
from core.pipeline.clusterization import ClusterizationPipeline
from core.clustering.KMeans import KmeansClusterModel
from core.embedding.ClipEmbedder import ClipEmbedder
import pandas as pd

parser = ArgumentParser()
parser.add_argument('folder')
parser.add_argument('-output_file', default='output.csv', required=False)

if __name__ == '__main__':
    args = parser.parse_args()
    pipeline = ClusterizationPipeline(ClipEmbedder(), KmeansClusterModel())
    names, result = pipeline.cluster(args.folder)
    pd.DataFrame({"img_name": names, "cluster_id": result}).to_csv(args.output_file, index=False)
