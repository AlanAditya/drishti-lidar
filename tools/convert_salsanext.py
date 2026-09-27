"""
Convert the pretrained SalsaNext (SemanticKITTI) PyTorch checkpoint to MLX.

Writes, next to the checkpoint:
  salsanext.safetensors         weights in MLX layout, for the C++ model (AdityaIntelligenceProMax/AI/SalsaNext.h)
  salsanext_golden.safetensors  one input and PyTorch's output for it, to check the C++ port against

and checks, before writing anything, that a Python MLX version of the network (the same ops, in
the same order, as the C++ port) reproduces PyTorch's output.

Usage (miniconda base env, which has torch + mlx):
  /opt/miniconda3/bin/python convert_salsanext.py /Users/adityadude/Documents/C++XCODE/pretrained

Layout changes (PyTorch -> MLX):
  - activations NCHW -> NHWC
  - conv weights [out, in, kh, kw] -> [out, kh, kw, in]
  - BatchNorm (eval) -> one per-channel scale + shift:  y = x * scale + shift,
      scale = weight / sqrt(running_var + eps),  shift = bias - running_mean * scale.
    It can't be folded into the conv before it: SalsaNext applies LeakyReLU between the two.
"""
import io
import pickletools
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

BN_EPS = 1e-5          # nn.BatchNorm2d default
LEAKY_SLOPE = 0.01     # nn.LeakyReLU default


# ------------------------------------------------------------------------------------------------
# Reference model: SalsaNext.py from github.com/TiagoCortinhal/SalsaNext (MIT), unchanged apart from
# dropping its training-only imports. Dropout is a no-op in eval mode.
# ------------------------------------------------------------------------------------------------
class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))
        resA1 = self.bn1(self.act2(self.conv2(shortcut)))
        resA2 = self.bn2(self.act3(self.conv3(resA1)))
        return shortcut + resA2


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super().__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        if pooling:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        shortcut = self.act1(self.conv1(x))
        resA1 = self.bn1(self.act2(self.conv2(x)))
        resA2 = self.bn2(self.act3(self.conv3(resA1)))
        resA3 = self.bn3(self.act4(self.conv4(resA2)))
        resA = self.bn4(self.act5(self.conv5(torch.cat((resA1, resA2, resA3), dim=1))))
        resA = shortcut + resA
        resB = self.dropout(resA) if self.drop_out else resA
        if self.pooling:
            return self.pool(resB), resA
        return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super().__init__()
        self.drop_out = drop_out
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.conv1 = nn.Conv2d(in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)
        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)
        upE1 = self.bn1(self.act1(self.conv1(upB)))
        upE2 = self.bn2(self.act2(self.conv2(upE1)))
        upE3 = self.bn3(self.act3(self.conv3(upE2)))
        upE = self.bn4(self.act4(self.conv4(torch.cat((upE1, upE2, upE3), dim=1))))
        if self.drop_out:
            upE = self.dropout3(upE)
        return upE


class SalsaNext(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)
        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)
        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        d = self.downCntx3(self.downCntx2(self.downCntx(x)))
        down0c, down0b = self.resBlock1(d)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)
        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        return F.softmax(self.logits(up1e), dim=1)


# ------------------------------------------------------------------------------------------------
# Loading the checkpoint safely
# ------------------------------------------------------------------------------------------------
# The checkpoint is PyTorch's legacy (non-zip) pickle and also stores the optimizer and LR
# scheduler, so torch.load(weights_only=True) refuses it. These are the only globals it was
# audited to reference (torch / numpy / container types, plus getattr(scheduler,
# "_triangular_scale_fn")). Anything else means it isn't the file we audited: stop before unpickling.
ALLOWED_GLOBALS = {
    "__builtin__.dict", "__builtin__.getattr", "_codecs.encode",
    "collections.OrderedDict", "collections.defaultdict",
    "numpy.core.multiarray.scalar", "numpy.dtype",
    "torch.DoubleStorage", "torch.FloatStorage", "torch.LongStorage",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_tensor_v2",
    "torch.optim.lr_scheduler.CyclicLR", "torch.optim.sgd.SGD",
}


def pickle_globals(path):
    data = io.BytesIO(Path(path).read_bytes())
    found = set()
    for _ in range(5):  # legacy format: magic, protocol, sys_info, the object, storage keys
        strings = []
        for op, arg, _pos in pickletools.genops(data):
            if op.name == "GLOBAL":
                found.add(arg.replace(" ", "."))
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                strings.append(arg)
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                found.add(strings[-2] + "." + strings[-1])
            if op.name == "STOP":
                break
    return found


def load_state_dict(checkpoint):
    unexpected = pickle_globals(checkpoint) - ALLOWED_GLOBALS
    if unexpected:
        sys.exit(f"refusing to unpickle {checkpoint}: unexpected globals {sorted(unexpected)}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}


# ------------------------------------------------------------------------------------------------
# PyTorch -> MLX weights
# ------------------------------------------------------------------------------------------------
def convert(state_dict):
    out = {}
    for name, t in state_dict.items():
        if name.endswith("num_batches_tracked"):
            continue
        prefix, leaf = name.rsplit(".", 1)
        module = prefix.rsplit(".", 1)[-1]
        a = t.detach().float().numpy()
        if module.startswith("bn"):
            if leaf != "weight":
                continue  # handled with the weight below
            scale = a / np.sqrt(state_dict[prefix + ".running_var"].numpy() + BN_EPS)
            shift = state_dict[prefix + ".bias"].numpy() - state_dict[prefix + ".running_mean"].numpy() * scale
            out[prefix + ".scale"] = scale.astype(np.float32)
            out[prefix + ".shift"] = shift.astype(np.float32)
        elif leaf == "weight":
            out[name] = np.ascontiguousarray(a.transpose(0, 2, 3, 1))  # [O,I,kh,kw] -> [O,kh,kw,I]
        else:
            out[name] = a
    return {k: mx.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------
# The same network in MLX (NHWC), op for op what AI/SalsaNext.h does in C++
# ------------------------------------------------------------------------------------------------
class MLXSalsaNext:
    def __init__(self, w):
        self.w = w

    def conv(self, x, name, padding=0, dilation=1):
        y = mx.conv2d(x, self.w[name + ".weight"], stride=1, padding=padding, dilation=dilation)
        return y + self.w[name + ".bias"]

    def bn(self, x, name):
        return x * self.w[name + ".scale"] + self.w[name + ".shift"]

    @staticmethod
    def leaky(x):
        return mx.maximum(x, LEAKY_SLOPE * x)

    @staticmethod
    def avg_pool3_s2(x):
        # nn.AvgPool2d(3, stride=2, padding=1), count_include_pad=True: always divides by 9.
        # As a depthwise conv with a constant 1/9 kernel.
        c = x.shape[-1]
        k = mx.full((c, 3, 3, 1), 1.0 / 9.0, dtype=x.dtype)
        return mx.conv2d(x, k, stride=2, padding=1, groups=c)

    @staticmethod
    def pixel_shuffle2(x):
        # NCHW PixelShuffle(2): out[c, 2h+i, 2w+j] = in[4c + 2i + j, h, w]; here in NHWC.
        n, h, w, c4 = x.shape
        c = c4 // 4
        x = x.reshape(n, h, w, c, 2, 2)           # channel 4c+2i+j -> (c, i, j)
        x = x.transpose(0, 1, 4, 2, 5, 3)          # (n, h, i, w, j, c)
        return x.reshape(n, h * 2, w * 2, c)

    def res_context(self, x, p):
        shortcut = self.leaky(self.conv(x, p + ".conv1"))
        a1 = self.bn(self.leaky(self.conv(shortcut, p + ".conv2", padding=1)), p + ".bn1")
        a2 = self.bn(self.leaky(self.conv(a1, p + ".conv3", padding=2, dilation=2)), p + ".bn2")
        return shortcut + a2

    def res_block(self, x, p, pooling):
        shortcut = self.leaky(self.conv(x, p + ".conv1"))
        a1 = self.bn(self.leaky(self.conv(x, p + ".conv2", padding=1)), p + ".bn1")
        a2 = self.bn(self.leaky(self.conv(a1, p + ".conv3", padding=2, dilation=2)), p + ".bn2")
        a3 = self.bn(self.leaky(self.conv(a2, p + ".conv4", padding=1, dilation=2)), p + ".bn3")
        a = self.bn(self.leaky(self.conv(mx.concatenate([a1, a2, a3], axis=-1), p + ".conv5")), p + ".bn4")
        a = shortcut + a
        return (self.avg_pool3_s2(a), a) if pooling else a

    def up_block(self, x, skip, p):
        b = mx.concatenate([self.pixel_shuffle2(x), skip], axis=-1)
        e1 = self.bn(self.leaky(self.conv(b, p + ".conv1", padding=1)), p + ".bn1")
        e2 = self.bn(self.leaky(self.conv(e1, p + ".conv2", padding=2, dilation=2)), p + ".bn2")
        e3 = self.bn(self.leaky(self.conv(e2, p + ".conv3", padding=1, dilation=2)), p + ".bn3")
        return self.bn(self.leaky(self.conv(mx.concatenate([e1, e2, e3], axis=-1), p + ".conv4")), p + ".bn4")

    def __call__(self, x):
        d = self.res_context(x, "downCntx")
        d = self.res_context(d, "downCntx2")
        d = self.res_context(d, "downCntx3")
        down0c, down0b = self.res_block(d, "resBlock1", True)
        down1c, down1b = self.res_block(down0c, "resBlock2", True)
        down2c, down2b = self.res_block(down1c, "resBlock3", True)
        down3c, down3b = self.res_block(down2c, "resBlock4", True)
        down5c = self.res_block(down3c, "resBlock5", False)
        up = self.up_block(down5c, down3b, "upBlock1")
        up = self.up_block(up, down2b, "upBlock2")
        up = self.up_block(up, down1b, "upBlock3")
        up = self.up_block(up, down0b, "upBlock4")
        return mx.softmax(self.conv(up, "logits"), axis=-1)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    arch = yaml.safe_load((folder / "arch_cfg.yaml").read_text())
    data = yaml.safe_load((folder / "data_cfg.yaml").read_text())
    sensor = arch["dataset"]["sensor"]
    H, W = sensor["img_prop"]["height"], sensor["img_prop"]["width"]
    nclasses = len(data["learning_map_inv"])

    state_dict = load_state_dict(folder / "SalsaNext")
    print(f"checkpoint: {len(state_dict)} tensors, {nclasses} classes, input {H}x{W}x5")

    # Golden pair: a normalised-range-image-like input (zero mean, unit variance per channel).
    torch.manual_seed(0)
    x = torch.randn(1, 5, H, W)
    model = SalsaNext(nclasses)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        ref = model(x).permute(0, 2, 3, 1).contiguous().numpy()  # NHWC [1, H, W, classes]

    weights = convert(state_dict)
    x_nhwc = mx.array(x.permute(0, 2, 3, 1).contiguous().numpy())
    out = MLXSalsaNext(weights)(x_nhwc)
    mx.eval(out)
    diff = float(np.abs(np.array(out) - ref).max())
    agree = float((np.array(out).argmax(-1) == ref.argmax(-1)).mean())
    print(f"MLX vs PyTorch: max |diff| {diff:.2e}, argmax agreement {agree * 100:.3f}%")
    if diff > 1e-3:
        sys.exit("MLX port doesn't match PyTorch - not writing anything")

    metadata = {
        "height": str(H), "width": str(W), "classes": str(nclasses),
        "fov_up": str(sensor["fov_up"]), "fov_down": str(sensor["fov_down"]),
        "img_means": ",".join(str(v) for v in sensor["img_means"]),
        "img_stds": ",".join(str(v) for v in sensor["img_stds"]),
        "learning_map_inv": ",".join(str(data["learning_map_inv"][i]) for i in range(nclasses)),
        "class_names": ",".join(data["labels"][data["learning_map_inv"][i]] for i in range(nclasses)),
        # SemanticKITTI's standard colour per class (data_cfg.yaml color_map is BGR), as RGB 0-255.
        "class_colors_rgb": ",".join(
            ",".join(str(c) for c in reversed(data["color_map"][data["learning_map_inv"][i]])) for i in range(nclasses)),
        "layout": "NHWC; conv [out,kh,kw,in]; bn as scale/shift",
    }
    mx.save_safetensors(str(folder / "salsanext.safetensors"), weights, metadata=metadata)
    mx.save_safetensors(str(folder / "salsanext_golden.safetensors"),
                        {"input": x_nhwc, "output": mx.array(ref)})
    print("wrote", folder / "salsanext.safetensors", "and", folder / "salsanext_golden.safetensors")


if __name__ == "__main__":
    main()
