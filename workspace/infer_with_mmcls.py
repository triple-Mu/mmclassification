import argparse
from pathlib import Path

from mmcls.apis import ImageClassificationInferencer

# training command:
# one card:
# python tools/train.py configs/heywhale/resnet50.py
#
# multiple cards:
# bash tools/dist_train.sh configs/heywhale/resnet50 4

args = argparse.Namespace()
args.config = '../configs/heywhale/resnet50.py'
args.checkpoint = '../work_dirs/resnet50/best_accuracy_top1_epoch_5.pth'
args.img_path = '../workspace/test_images'
args.vis_path = '../workspace/test_vis'
args.device = 'mps'

inferencer = ImageClassificationInferencer(
    args.config, args.checkpoint, device=args.device)
inferencer.show_progress = False

for img_path in Path(args.img_path).iterdir():
    result = inferencer(img_path, show=False, show_dir=args.vis_path)[0]
    print(f'{img_path.name:<42s}: '
          f'pred_class: {result["pred_class"]} '
          f'pred_score: {result["pred_score"]:.4f}')
