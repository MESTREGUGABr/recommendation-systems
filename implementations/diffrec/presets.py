"""Settings from the authors' checkpoint names and reference scores from the paper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    lr: float
    hidden: tuple[int, ...]
    steps: int
    scale: float
    noise_min: float
    noise_max: float
    reference: tuple[float, ...]
    table: int
    reweight: bool = True
    batch_size: int = 400
    ae_lr: float | None = None
    lamda: float = 0.03
    time_min: float | None = None


MODELS = {
    "diffrec": {
        "amazon-book_clean": Preset(5e-5, (1000,), 5, .0001, .0005, .005, (.0695, .1010, .0451, .0547), 2),
        "yelp_clean": Preset(1e-5, (1000,), 5, .01, .001, .01, (.0581, .0960, .0363, .0478), 2),
        "ml-1m_clean": Preset(.001, (200, 600), 40, .005, .005, .01, (.1058, .1787, .0901, .1148), 2),
        "amazon-book_noisy": Preset(5e-5, (1000,), 10, .005, .0001, .0005, (.0546, .0822, .0335, .0419), 3),
        "yelp_noisy": Preset(1e-5, (1000,), 5, .001, .0005, .01, (.0507, .0853, .0309, .0414), 3, reweight=False),
        "ml-1m_noisy": Preset(.001, (200, 600), 5, .5, .001, .01, (.0658, .1236, .0488, .0703), 3, reweight=False),
    },
    "l-diffrec": {
        "amazon-book_clean": Preset(.0001, (300,), 5, .5, .001, .005, (.0694, .1028, .0440, .0540), 6, ae_lr=.0005, lamda=.05),
        "yelp_clean": Preset(.0001, (300,), 5, .01, .005, .01, (.0585, .0970, .0353, .0469), 6, ae_lr=.0005, reweight=False),
        "ml-1m_clean": Preset(.0005, (300,), 40, .005, .005, .02, (.1060, .1809, .0868, .1122), 6, ae_lr=.001),
        "amazon-book_noisy": Preset(.0001, (300,), 5, .5, .001, .005, (.0586, .0876, .0347, .0434), 4, ae_lr=.0005, lamda=.07),
        "yelp_noisy": Preset(.0001, (300,), 5, .01, .005, .01, (.0521, .0876, .0311, .0419), 4, ae_lr=.0005, reweight=False),
        "ml-1m_noisy": Preset(.0005, (300,), 100, .005, .005, .02, (.0665, .1272, .0493, .0710), 4, ae_lr=.001),
    },
    "t-diffrec": {
        "amazon-book_clean": Preset(1e-5, (1000,), 10, .0005, .001, .005, (.0819, .1139, .0565, .0661), 7, time_min=.1),
        "yelp_clean": Preset(1e-5, (1000,), 5, .005, .001, .01, (.0601, .0987, .0377, .0494), 7, time_min=.5),
    },
    "lt-diffrec": {
        "amazon-book_clean": Preset(.0001, (300,), 5, .7, .001, .005, (.0838, .1188, .0560, .0665), 7, ae_lr=.0005, time_min=.1),
        "yelp_clean": Preset(5e-5, (300,), 5, .01, .005, .01, (.0604, .0982, .0369, .0484), 7, batch_size=200, ae_lr=.0005, time_min=.5),
    },
}

METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
