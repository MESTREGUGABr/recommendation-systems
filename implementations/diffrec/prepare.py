"""Fetch the authors' pinned code and extract their original data splits."""

import argparse
import hashlib
from pathlib import Path
from shutil import copyfileobj
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
DATASETS = ROOT / "datasets"
REPOSITORY = "https://github.com/YiyanXu/DiffRec.git"
REVISION = "e42532f5f329a5f1c5230f05ad7e2930d77b9f75"
AMAZON_EMBEDDINGS_URL = "https://pan.ustc.edu.cn/seafile/f/2a4185182c9940c18e7f/?dl=1"
AMAZON_EMBEDDINGS = {
    "amazon-book_clean": "b9368edf3264a6ee9dff6fed4be83a3402789cf683ef01e774a8d929f92e95a1",
    "amazon-book_noisy": "f085ac9097e32f0731c56a1501e619e1d2002208f2cc7ff99ee99fbe99bee3a8",
}
ARCHIVES = {
    "amazon-book_clean": "c11b68372f795540eb9e93270c17dafe09d0a60499276a2185e63a42f110be48",
    "amazon-book_noisy": "1f5238b0cf617b16911e5b5ecdade529566bb4a1bd7941daf02d7dd1e55f1c19",
    "ml-1m_clean": "407d6f1deac6e373c3a7157ce9cf974fcd42f43967d7f0cfaa25b01bc2521050",
    "ml-1m_noisy": "45123814e37198274f525b8f79711b97937a99077515c6cb5fc1c6cc6aff8fae",
    "yelp_clean": "32d02392c0b89bc87ee6ee6342138d472b2db633ad28cf0d45874942d09129f8",
    "yelp_noisy": "590fa3bad26530c0fa6daf59a15964893767ad1509335e0af97b55b2abf935b1",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(UPSTREAM), *args], text=True).strip()


def verify_upstream():
    if not (UPSTREAM / ".git").is_dir():
        raise ValueError("Código oficial ausente. Execute python prepare.py.")
    if git("rev-parse", "HEAD") != REVISION or git("status", "--porcelain", "-uno"):
        raise ValueError(f"A cópia oficial deve estar limpa na revisão {REVISION}.")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_amazon_embeddings(names, downloaded_zip):
    missing = []
    for name in names:
        if name in AMAZON_EMBEDDINGS:
            target = DATASETS / name / "item_emb.npy"
            if not target.exists() or sha256(target) != AMAZON_EMBEDDINGS[name]:
                missing.append(name)
    if not missing:
        return

    with TemporaryDirectory(dir=DATASETS) as temporary:
        work = Path(temporary)
        archive = downloaded_zip or work / "download.zip"
        if downloaded_zip is None:
            print("Baixando embeddings oficiais de Amazon-book...", flush=True)
            with urlopen(AMAZON_EMBEDDINGS_URL, timeout=60) as response, archive.open("wb") as output:
                copyfileobj(response, output)
        # The file-share service wraps the authors' uploaded ZIP in a second ZIP.
        inner = work / "embeddings.zip"
        with ZipFile(archive) as outer:
            if "DiffRec/amazon_item_emb.zip" in outer.namelist():
                with outer.open("DiffRec/amazon_item_emb.zip") as source, inner.open("wb") as output:
                    copyfileobj(source, output)
            else:
                inner = archive
        with ZipFile(inner) as embeddings:
            for name in missing:
                with embeddings.open(f"amazon_item_emb/{name}/item_emb.npy") as source:
                    staged = work / "item_emb.npy"
                    with staged.open("wb") as output:
                        copyfileobj(source, output)
                if sha256(staged) != AMAZON_EMBEDDINGS[name]:
                    raise ValueError(f"SHA-256 incorreto nos embeddings de {name}.")
                staged.replace(DATASETS / name / "item_emb.npy")
                print(f"{name}: embeddings oficiais disponíveis.")


def prepare(names, amazon_zip=None):
    if not UPSTREAM.exists():
        with TemporaryDirectory(prefix=".prepare-", dir=ROOT) as temporary:
            checkout = Path(temporary) / "source"
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", REPOSITORY, REVISION], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", REVISION], check=True)
            checkout.rename(UPSTREAM)
    verify_upstream()

    DATASETS.mkdir(exist_ok=True)
    for name in names:
        archive = UPSTREAM / "datasets" / f"{name}.rar"
        if sha256(archive) != ARCHIVES[name]:
            raise ValueError(f"SHA-256 incorreto em {archive.name}.")

        files = ["train_list.npy", "valid_list.npy", "test_list.npy"]
        if not name.startswith("amazon-book_"):
            files.append("item_emb.npy")
        with TemporaryDirectory(dir=DATASETS) as temporary:
            members = [f"{name}/{filename}" for filename in files]
            subprocess.run(["tar", "-xf", str(archive), "-C", temporary, *members], check=True)
            destination = DATASETS / name
            destination.mkdir(exist_ok=True)
            for filename in files:
                (Path(temporary) / name / filename).replace(destination / filename)
        print(f"{name}: partições oficiais disponíveis em {destination}")
    prepare_amazon_embeddings(names, amazon_zip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help=", ".join(ARCHIVES))
    parser.add_argument("--amazon-embeddings", type=Path, help="ZIP oficial de embeddings já baixado")
    args = parser.parse_args()
    names = args.datasets or list(ARCHIVES)
    if unknown := set(names) - ARCHIVES.keys():
        parser.error(f"Datasets desconhecidos: {', '.join(sorted(unknown))}")
    try:
        prepare(names, args.amazon_embeddings)
    except (ValueError, OSError, KeyError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Falha ao preparar DiffRec: {exc}\n")
