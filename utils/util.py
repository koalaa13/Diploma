import os
import shutil

src_dir = '../estimator/estimator_generated_embeddings'
dst_dir = '../data/nn_embedding_transformed'

for file in os.listdir(src_dir):
    acc = float(file[file.find('_') + 1:file.rfind('.')])
    if acc > 80.0:
        shutil.copy(os.path.join(src_dir, file), dst_dir)
i = 0
for file in os.listdir(dst_dir):
    os.rename(os.path.join(dst_dir, file), os.path.join(dst_dir, str(i) + '.emb'))
    i += 1
