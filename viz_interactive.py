#!/usr/bin/env python
"""Interactive completion visualization (rotatable popup windows).

For one test sample it opens ONE matplotlib window with two 3D panels:
  LEFT  : object point cloud, colored RED->GREEN by contact/closeness (big=red)
  RIGHT : model-completed hand, each point colored by its part/link label (ch3)

Usage (run on a machine with a display; you have DISPLAY=:0):
  micromamba activate test-py312
  python viz_interactive.py --sample 5
  python viz_interactive.py --samples 0 5 20
  python viz_interactive.py --sample 5 --save-only out/   # headless PNGs

Mouse: left-drag rotate, right-drag/scroll zoom. Close the window to move on.
"""
import os, argparse, time
import numpy as np
import torch

# NOTE: import order matters. train_completion pulls in utils.visualize which
# calls matplotlib.use('agg') at import time and would override any GUI backend
# we set first. So we import it first, then force Qt5Agg / TkAgg afterwards.
import train_completion as T

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _pick_backend():
    for b in ("Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(b, force=True)   # switch even though pyplot already imported
            return b
        except Exception:
            continue
    return "Agg"

print(f"[backend] {_pick_backend()}")

# part label -> readable-ish name (5 links, correspond to the 5 stored parts 1..5)
PART_NAMES = {1: "link1", 2: "link2", 3: "link3", 4: "link4", 5: "link5"}
PART_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red", 5: "tab:purple"}


def build_model(model_path, svpoints=50, nc=5, embed_dim=64):
    args = argparse.Namespace(svpoints=svpoints, nc=nc, embed_dim=embed_dim,
                              attention=True, dropout=0.1)
    betas = T.get_betas("linear", 0.0001, 0.02, 1000)
    model = T.Model(args, betas, "mse", "eps", "fixedsmall").cuda()
    model.eval()
    ckpt = torch.load(model_path, weights_only=False)
    sd = ckpt["model_state"]
    sd = {k.replace("model.module.", "model.", 1) if k.startswith("model.module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def draw_panel(ax, xyz, color, title, size=4, cmap_name=None, vmin=None, vmax=None, cb_label=None):
    if cmap_name:
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, cmap=cmap_name,
                        s=size, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label(cb_label or "")
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=size)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def visualize_sample(model, objs, idx, svpoints, save_only=None):
    obj = objs[idx]                     # (2048,5) file obj: xyz, parttag(ch3), closeness(ch4)
    xyz_obj = obj[:, :3].float().numpy()
    closeness = obj[:, 4].float().numpy()

    # ---- build the model condition (ch4 -> class 0, as Rdataset did at train) ----
    partial = obj[:svpoints].clone()    # first svpoints = nearest-to-hand object pts
    partial[:, 4] = 0.0                 # last channel = class label 0
    partial_x = partial.transpose(0, 1).unsqueeze(0).cuda()   # [1,5,svpoints]
    gen_shape = (1, 5, 300 - svpoints)  # (B, nc, points-to-generate)
    with torch.no_grad():
        recon = model.gen_samples(partial_x, gen_shape, device="cuda", clip_denoised=False)
    recon = recon.transpose(1, 2).cpu()          # [1,300,5]

    # completed hand = the generated (non-context) region; color by predicted part label
    hand = recon[0, svpoints:]                   # (250,5)
    hand_xyz = hand[:, :3].numpy()
    hand_part = np.rint(hand[:, 3].numpy()).astype(int)
    hand_part = np.clip(hand_part, 1, 5)
    hand_colors = np.array([matplotlib.colors.to_rgb(PART_COLORS.get(p, "gray")) for p in hand_part])

    # ---- figure 1: object point cloud alone (standard view, colored by contact) ----
    fig_obj = plt.figure(figsize=(9, 8))
    axo = fig_obj.add_subplot(1, 1, 1, projection="3d")
    oso = axo.scatter(xyz_obj[:, 0], xyz_obj[:, 1], xyz_obj[:, 2], c=closeness,
                      cmap="RdYlGn_r", s=8, alpha=0.8, depthshade=False)
    cbo = fig_obj.colorbar(oso, ax=axo, shrink=0.6, pad=0.08)
    cbo.set_label("object contact closeness (red=high, green=low)")
    axo.set_title(f"sample #{idx} : object point cloud")
    axo.set_box_aspect((1, 1, 1))
    axo.set_xlabel("x"); axo.set_ylabel("y"); axo.set_zlabel("z")

    # ---- figure 2: object + completed hand in ONE 3D view; hand markers enlarged ----
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    # object: dense cloud, small markers, colored by contact (red=high, green=low)
    obj_sc = ax.scatter(xyz_obj[:, 0], xyz_obj[:, 1], xyz_obj[:, 2], c=closeness,
                        cmap="RdYlGn_r", s=3, alpha=0.5, depthshade=False)
    cb = fig.colorbar(obj_sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("object contact closeness (red=high, green=low)")
    # completed hand: sparse cloud -> large markers, colored by part/link
    hand_sc = ax.scatter(hand_xyz[:, 0], hand_xyz[:, 1], hand_xyz[:, 2], c=hand_colors,
                         s=150, alpha=1.0, depthshade=False, edgecolors="black", linewidths=0.3)
    ax.set_title(f"sample #{idx}  : object (contact colormap) + completed hand (part colors)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # legend for hand part colors
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PART_COLORS[p],
                          markersize=9, label=PART_NAMES[p]) for p in range(1, 6)]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    if save_only:
        os.makedirs(save_only, exist_ok=True)
        p1 = os.path.join(save_only, f"sample{idx:04d}_obj.png")
        fig_obj.savefig(p1, dpi=130)
        p2 = os.path.join(save_only, f"sample{idx:04d}_combo.png")
        fig.savefig(p2, dpi=130)
        plt.close(fig_obj); plt.close(fig)
        print("saved", p1, "and", p2)
        return
    else:
        print(f"\n[window] sample #{idx}: two figures opened - close BOTH to move on.")
        plt.show()
        plt.close(fig_obj); plt.close(fig)
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="checkpoint path (epoch_*.pth)")
    ap.add_argument("--pt", default="datasets/manohand.pt")
    ap.add_argument("--svpoints", type=int, default=50)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--samples", type=int, nargs="*", default=[0])
    ap.add_argument("--save-only", default=None)
    ap.add_argument("--seed", type=int, default=42)
    opt = ap.parse_args()

    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    samples = [opt.sample] if opt.sample is not None else opt.samples
    if not opt.model:
        raise SystemExit("error: --model <checkpoint.pth> is required")

    print("loading objs ...")
    t0 = time.time()
    d = torch.load(opt.pt, weights_only=False, map_location="cpu")
    objs = d["objs"]   # float32 (31292,2048,5)
    print(f"  loaded in {time.time()-t0:.1f}s  objs{tuple(objs.shape)}")

    model = build_model(opt.model, svpoints=opt.svpoints)
    print("model ready.")

    for i in samples:
        visualize_sample(model, objs, i, opt.svpoints, save_only=opt.save_only)


if __name__ == "__main__":
    main()
