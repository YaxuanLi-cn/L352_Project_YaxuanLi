import torch
from gh import metrics as M
CHARB_EPS = 1e-3
NEW_METRICS = {"lpips_squeeze": dict(sign=-1, family="lpips", label="LPIPS-Squeeze"),
               "charbonnier": dict(sign=-1, family="pixel", label="Charbonnier"),
               "haarpsi": dict(sign=1, family="haarpsi", label="HaarPSI")}
NEW_LOSSES = {"lpips_squeeze": "lpips_squeeze", "charbonnier": "charbonnier", "dreamsim": "dreamsim", "vif": "vif",
              "haarpsi": "haarpsi"}
NEW_LOSS_LABEL = {"lpips_squeeze": "LPIPS-Squeeze", "charbonnier": "Charbonnier", "dreamsim": "DreamSim", "vif": "VIF",
                  "haarpsi": "HaarPSI"}
_orig_loss_form = M.loss_form


def loss_form(metric, value):
    if metric == "haarpsi":
        return 1.0 - value
    if metric in NEW_METRICS:
        return value
    return _orig_loss_form(metric, value)


def install():
    M.LOSSES.update(NEW_LOSSES)
    M.LOSS_LABEL.update(NEW_LOSS_LABEL)
    for k, v in NEW_METRICS.items():
        M.SIGN[k], M.FAMILY[k], M.LABEL[k] = v["sign"], v["family"], v["label"]
    M.loss_form = loss_form


class Metrics(M.Metrics):
    def _net(self, key):
        if key == "lpips_squeeze":
            if key not in self._nets:
                import lpips
                net = lpips.LPIPS(net="squeeze", verbose=False).to(self.device).eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self._nets[key] = net
            return self._nets[key]
        return super()._net(key)
    def __call__(self, name, t, r):
        if name == "lpips_squeeze":
            return self._net(name)(t, r, normalize=True).reshape(-1)
        if name == "charbonnier":
            return (torch.sqrt((t - r) ** 2 + CHARB_EPS ** 2) - CHARB_EPS).mean(dim=(1, 2, 3))
        if name == "haarpsi":
            import piq
            return piq.haarpsi(t.clamp(0, 1), r.clamp(0, 1), data_range=1.0, reduction="none")
        return super().__call__(name, t, r)
