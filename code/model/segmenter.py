"""
Person silhouettes with SAM 2.1 (Meta, 2024; Apache-2.0 licence).

MediaPipe's own person mask is soft and blobby around the arms and clothes.
SAM 2 is prompted with the person's box and a few body points from MediaPipe
and returns a much sharper mask — closer to the clean BodyM silhouettes the
hybrid model was trained on.

Weights are downloaded from Hugging Face on first use (~320 MB):
  facebook/sam2.1-hiera-base-plus

Tested 2026-10-19 on the 24 real-photo people (scripts/run_models_on_datasets.py
--sam2): masks are visibly sharper, but the hybrid model's errors did not improve
(waist slightly worse), so the app keeps MediaPipe masks. Re-test with your own
shop data before switching it on.
"""

import numpy as np
import torch

CHECKPOINT = 'facebook/sam2.1-hiera-base-plus'


class PersonSegmenter:
    def __init__(self, device=None):
        from transformers import Sam2Model, Sam2Processor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Sam2Processor.from_pretrained(CHECKPOINT)
        self.model = Sam2Model.from_pretrained(CHECKPOINT).to(self.device).eval()

    @torch.no_grad()
    def mask(self, img_bgr, kps33, min_vis=0.3):
        """
        img_bgr: photo; kps33: MediaPipe keypoints [(x_px, y_px, vis)] * 33.
        Returns an H x W float mask (0/1) of the person, or None.
        """
        k = np.asarray(kps33, np.float32)
        vis = k[k[:, 2] > min_vis]
        if len(vis) < 6:
            return None
        h, w = img_bgr.shape[:2]
        x0, y0 = vis[:, :2].min(0)
        x1, y1 = vis[:, :2].max(0)
        mx, my = 0.15 * (x1 - x0) + 10, 0.10 * (y1 - y0) + 10   # keypoints stop short of skin/hair
        box = [float(max(0, x0 - mx)), float(max(0, y0 - 1.5 * my)),
               float(min(w - 1, x1 + mx)), float(min(h - 1, y1 + my))]
        # positive clicks on the torso and legs, where the body certainly is
        pts = []
        for a, b in ((11, 12), (23, 24), (25, 26)):          # shoulders, hips, knees
            if k[a, 2] > min_vis and k[b, 2] > min_vis:
                pts.append([float((k[a, 0] + k[b, 0]) / 2), float((k[a, 1] + k[b, 1]) / 2)])
        rgb = img_bgr[:, :, ::-1].copy()
        inputs = self.processor(images=rgb, input_boxes=[[box]],
                                input_points=[[pts]] if pts else None,
                                input_labels=[[[1] * len(pts)]] if pts else None,
                                return_tensors='pt').to(self.device)
        out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'])[0]
        return (masks.reshape(-1, h, w)[0].numpy() > 0.5).astype(np.float32)
