#!/usr/bin/env python
"""manohand point-cloud completion -- inference.

Loads a saved completion checkpoint, runs the diffusion reverse process on a few
held-out test hands, saves the completed-hand cloud together with the GT hand,
and reports per-sample Chamfer distance.

The diffusion/Model classes are shared with train_completion (see T.Model).
"""
import os
import argparse
import numpy as np
import torch

import train_completion as T
from datasets.Rdataset import Rdataset


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance between two unordered point sets."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    da = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min(-1).mean()
    db = np.sqrt(((b[:, None] - a[None]) ** 2).sum(-1)).min(-1).mean()
    return float((da + db) / 2)


def main(opt):
    torch.manual_seed(opt.seed); np.random.seed(opt.seed)

    args = argparse.Namespace(svpoints=opt.svpoints, nc=opt.nc,
                              embed_dim=opt.embed_dim, attention=True, dropout=0.1)
    betas = T.get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = T.Model(args, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    ckpt = torch.load(opt.model, weights_only=False)
    sd = ckpt["model_state"]
    sd = {k.replace("model.module.", "model.", 1) if k.startswith("model.module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd)
    print("loaded:", opt.model)

    ds = Rdataset(root=opt.dataroot, category=opt.category, split="test",
                  testnum=opt.testnum, mode="cmap", npoints=opt.npoints,
                  sv_samples=opt.svpoints)
    os.makedirs(opt.outdir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for i in opt.samples:
            item = ds[i]
            gt = item["train_points"][:, :3].numpy()               # (300,3) GT hand
            sv = item["sv_points"].transpose(1, 2)                 # (1,5,300)
            partial = sv[:, :, :opt.svpoints].cuda()
            gen_shape = (1, opt.nc, opt.npoints - opt.svpoints)
            recon = model.gen_samples(partial, gen_shape, device="cuda",
                                      clip_denoised=False).cpu()
            recon = recon.transpose(1, 2)                          # (1,300,5)
            completed = recon[0, opt.svpoints:, :3].numpy()        # generated hand region
            cd = chamfer(gt[opt.svpoints:], completed)             # compare hand regions
            np.savez(os.path.join(opt.outdir, f"sample{i:04d}.npz"),
                     gt_hand=gt, completed_hand=completed, index=i, chamfer=cd)
            rows.append((i, cd))
            print(f"sample {i:3d}: Chamfer={cd:.4f}  -> {os.path.join(opt.outdir, f'sample{i:04d}.npz')}")

    print("\n=== Chamfer(GT hand, completed hand) ===")
    for i, cd in rows:
        print(f"  #{i:3d}: {cd:.4f}")
    if rows:
        print(f"mean = {np.mean([c for _, c in rows]):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint path (epoch_*.pth)")
    p.add_argument("--dataroot", default="datasets")
    p.add_argument("--category", nargs="*", default=["manohand.pt"])
    p.add_argument("--samples", type=int, nargs="*", default=[0, 5, 20])
    p.add_argument("--testnum", type=int, default=500, help="rows taken as test split")
    p.add_argument("--nc", type=int, default=5)
    p.add_argument("--npoints", type=int, default=300)
    p.add_argument("--svpoints", type=int, default=50)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--outdir", default="output/completion")
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--schedule_type", default="linear")
    p.add_argument("--time_num", type=int, default=1000)
    p.add_argument("--loss_type", default="mse")
    p.add_argument("--model_mean_type", default="eps")
    p.add_argument("--model_var_type", default="fixedsmall")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
