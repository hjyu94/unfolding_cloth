import os
import shutil
for i in range(1, 50):
    source_path = f"/home/hjeong/code/unfolding_cloth/datasets/cloth_competition_dataset_0000/sample_{'{0:06d}'.format(i)}/detected_edge/extracted_pcd.ply"
    destination_path = f"/home/hjeong/code/unfolding_cloth/graph/data/ICRA/raw/sample_{'{0:06d}'.format(i)}.ply"

    try:
        shutil.copy(source_path, destination_path)
        print(f'{source_path}에서 {destination_path}로 파일이 이동되었습니다.')
    except Exception as e:
        print(e)