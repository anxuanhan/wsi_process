"""
encode_text_queries.py
----------------------
Encode multi-class histology prompts with CONCH.

Example:
    python encode_text_queries.py \
        --checkpoint_path /scratchdata1/users/a1978372/Anxuan/CONCH/checkpoints/conch/pytorch_model.bin \
        --output_dir ./text_features_histology_regions

Outputs:
    class_features.npy        shape: (num_classes, 512), one averaged prototype per class
    prompt_features.npy       shape: (num_prompts, 512), one feature per prompt
    prompt_labels.npy         shape: (num_prompts,), class index for each prompt
    classes.txt               class names, one per line
    prompts.json              prompts grouped by class
    queries.txt               human-readable prompt list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CONCH_ROOT = REPO_ROOT / "CONCH"
if CONCH_ROOT.exists() and str(CONCH_ROOT) not in sys.path:
    sys.path.insert(0, str(CONCH_ROOT))

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
CLASSES = [
    "malignant tissue",
    "benign tissue",
    "stroma",
    "lymphocytes",
    "necrosis",
    "adipose tissue",
    "tissue artifact",
    "blood vessel",
    "extracellular mucin",
    "nerve",
    "hemorrhage",
    "smooth muscle",
    "plasma cells",
]


# CLASSES = [
#     "malignant tissue",
#     "benign tissue",
#     "stroma",
#     "lymphocytes",
#     "necrosis",
#     "adipose tissue",
#     "tissue artifact",
#     "blood vessel",
#     "extracellular mucin",
#     "nerve",
#     "hemorrhage",
#     "smooth muscle",
#     "plasma cells",
# ]

LABEL_TO_COLOR = {
    "malignant tissue": (217, 30, 30),
    "benign tissue": (60, 133, 194),
    "stroma": (250, 194, 99),
    "lymphocytes": (90, 186, 125),
    "necrosis": (128, 0, 128),
    "adipose tissue": (245, 237, 203),
    "tissue artifact": (128, 128, 128),
    "blood vessel": (255, 182, 193),
    "extracellular mucin": (0, 255, 255),
    "nerve": (75, 0, 130),
    "hemorrhage": (165, 42, 42),
    "smooth muscle": (210, 105, 30),
    "plasma cells": (255, 20, 147),
}

PROMPT_VERSION = "prostate_histology_13_region_specific_prompts_v2"

PROMPTS_BY_CLASS = {

  "malignant tissue": [
   "prostate adenocarcinoma composed of small discrete infiltrative glands",
    "closely packed small malignant prostate glands infiltrating surrounding stroma",
    "numerous separate small malignant acini with irregular crowded distribution",
    "small well-formed malignant glands infiltrating between larger benign prostate glands",
    "crowded malignant glands with relatively uniform small lumina and reduced intervening stroma",
    "irregular angulated malignant prostate glands distributed infiltratively through stroma",
    "poorly formed malignant glands with incomplete or irregular lumina",
    "fused malignant prostate glands forming confluent irregular glandular structures",
    "cribriform prostate adenocarcinoma with large complex epithelial structures and multiple irregular luminal spaces",
    "high-grade prostate adenocarcinoma with dense complex fused and cribriform glandular architecture",

],

"benign tissue": [
    "large regularly spaced benign prostate glands with smooth rounded contours",
    "benign prostate glands separated by abundant fibromuscular stroma",
    "well-circumscribed benign glands with uniform epithelial cells and bland nuclei",
    "benign prostate glands with prominent papillary epithelial infolding",
    "large benign prostatic acini containing abundant luminal secretions",
    "benign prostate glands containing corpora amylacea",
    "benign prostatic hyperplasia with large dilated regularly shaped glands",
    "normal prostate glands with orderly architecture and widely separated acini",
],

    "stroma": [
        "fibrous prostate stroma between glandular structures",
        "collagenous prostatic stroma with spindle fibroblast nuclei",
        "pink eosinophilic stromal collagen without epithelial glands",
        "loose fibrous prostate stroma with wavy collagen bundles",
        "stromal connective tissue dominated by extracellular collagen",
        "fibrovascular prostate stroma with sparse elongated stromal cells",
        "fibrous stroma separating benign or malignant prostatic glands",
        "reactive fibrous stroma with dense collagen and spindle stromal cells",
        "collagen-rich stromal tissue with scattered spindle cells",
        "prostate connective tissue lacking obvious epithelial structures",
    ],

    "lymphocytes": [
        "dense lymphocytic infiltrate of small dark round nuclei",
        "aggregates of lymphocytes in H&E stained tissue",
        "small monomorphic inflammatory lymphoid cells",
        "blue-purple lymphocyte-rich inflammatory infiltrate",
        "clusters of small mature lymphocytes with scant cytoplasm",
        "lymphoid aggregate composed of many tiny dark nuclei",
        "tumor infiltrating lymphocytes surrounding tissue structures",
        "inflammatory area dominated by lymphocytes",
        "dense collection of small round lymphocytes in stroma",
        "sheet of small dark lymphoid cells with minimal cytoplasm",
    ],

    "necrosis": [
        "necrotic tissue with ghost cells and eosinophilic debris",
        "acellular necrotic debris in H&E stained tissue",
        "coagulative necrosis with loss of nuclear detail",
        "dirty necrosis with granular cellular debris",
        "dead tissue containing karyorrhectic nuclear fragments",
        "pale necrotic region lacking viable cells",
        "eosinophilic necrotic material with fragmented nuclei",
        "area of tissue necrosis and cellular breakdown",
        "nonviable tissue with loss of normal architecture",
        "necrotic area containing amorphous eosinophilic debris",
    ],

    "adipose tissue": [
        "adipose tissue with large clear fat vacuoles",
        "white fat cells with thin cytoplasmic rims",
        "mature adipocytes forming empty round spaces",
        "periprostatic fatty tissue dominated by mature adipocytes",
        "clusters of large clear adipose cells in H&E",
        "fat tissue with delicate septa between adipocytes",
        "lipid vacuoles appearing as white empty spaces",
        "benign adipose tissue with sparse nuclei",
        "mature fat cells with peripheral flattened nuclei",
        "adipose tissue composed of large clear vacuolated cells",
    ],

    "tissue artifact": [
        "tissue processing artifact with folds or tears",
        "out-of-focus blurry histology artifact",
        "crushed tissue artifact with distorted cells",
        "section fold artifact in H&E stained tissue",
        "air bubble or scanning artifact obscuring tissue",
        "staining artifact with abnormal color or precipitate",
        "damaged tissue edge with mechanical artifact",
        "non-diagnostic tissue artifact rather than real histology",
        "histology region distorted by technical artifact",
        "processing artifact obscuring normal tissue morphology",
    ],

    "blood vessel": [
        "blood vessel lumen lined by endothelial cells",
        "vascular structure containing red blood cells",
        "artery or vein in H&E stained tissue",
        "round vessel lumen filled with erythrocytes",
        "endothelial-lined channel with intraluminal blood",
        "small capillary or venule within stromal tissue",
        "blood vessel with red blood cells in the lumen",
        "vascular lumen surrounded by connective tissue",
        "well-defined vascular channel containing erythrocytes",
        "endothelial-lined vascular structure within tissue",
    ],

    "extracellular mucin": [
        "extracellular mucin pools in H&E stained tissue",
        "pale blue-gray mucin outside tumor cells",
        "amorphous extracellular mucinous material",
        "mucin lakes with floating epithelial cells",
        "gelatinous extracellular mucin separating tissue fragments",
        "abundant mucin matrix with low cellularity",
        "mucinous stroma containing pale extracellular mucin",
        "acellular mucin pool in histopathology image",
        "pale basophilic mucinous material outside cells",
        "large extracellular mucin pool with scant cells",
    ],

    "nerve": [
        "peripheral nerve bundle with wavy spindle nuclei",
        "nerve fascicle in H&E stained tissue",
        "neural tissue with elongated Schwann cell nuclei",
        "pink nerve fibers arranged in a fascicular pattern",
        "perineurial sheath surrounding a nerve bundle",
        "wavy nerve fibers embedded in connective tissue",
        "peripheral nerve within prostatic stroma",
        "neural fascicle with parallel elongated nuclei",
        "well-defined nerve bundle with wavy fibers",
        "nerve tissue showing elongated nuclei and fascicular architecture",
    ],

    "hemorrhage": [
        "hemorrhage with abundant extravasated red blood cells",
        "blood-filled area outside vascular structures",
        "red blood cells spilling through tissue",
        "fresh hemorrhage in H&E stained section",
        "dense collection of erythrocytes in tissue space",
        "area dominated by red blood cells and blood clot",
        "extravasated blood surrounding histologic tissue",
        "hemorrhagic tissue with pools of erythrocytes",
        "accumulation of red blood cells outside vessels",
        "tissue space filled with extravasated erythrocytes",
    ],

    "smooth muscle": [
        "smooth muscle bundles with elongated cigar-shaped nuclei",
        "eosinophilic smooth muscle fibers in fascicular arrangement",
        "pink spindle cell smooth muscle tissue",
        "interlacing bundles of smooth muscle cells",
        "smooth muscle with uniform elongated nuclei and eosinophilic cytoplasm",
        "dense eosinophilic muscle fibers without epithelial glands",
        "fascicular smooth muscle tissue in H&E section",
        "well-formed smooth muscle bundles with parallel nuclei",
        "smooth muscle tissue composed of elongated spindle cells",
        "compact smooth muscle fascicles with cigar-shaped nuclei",
    ],

    "plasma cells": [
        "plasma cells with eccentric nuclei and perinuclear hof",
        "dense infiltrate of mature plasma cells",
        "inflammatory plasma cells with clock-face chromatin",
        "oval plasma cells with basophilic cytoplasm",
        "clusters of antibody-producing plasma cells",
        "plasmacytic infiltrate in H&E stained tissue",
        "eccentric nuclei and abundant cytoplasm in plasma cells",
        "chronic inflammation rich in plasma cells",
        "mature plasma cells with eccentric round nuclei",
        "plasma cell infiltrate with perinuclear clearing",
    ],

}





## 我们自己写的prompt
# PROMPTS_BY_CLASS = {
#     "malignant tissue": [
#         "prostate adenocarcinoma composed of infiltrative malignant glands",
#         "Gleason pattern 3 prostate cancer with separate crowded small glands",
#         "Gleason pattern 4 prostate cancer with fused and poorly formed glands",
#         "cribriform prostate adenocarcinoma with confluent epithelial proliferation",
#         "Gleason pattern 5 prostate cancer with solid sheets or single tumor cells",
#         "small malignant prostatic glands lacking a basal cell layer",
#         "atypical prostate glands with enlarged nuclei and prominent nucleoli",
#         "invasive prostatic adenocarcinoma growing through fibromuscular stroma",
#         "malignant prostate epithelium with disorganized glandular architecture",
#         "prostate carcinoma showing perineural invasion around a nerve bundle",
#     ],
#     "benign tissue": [
#         "benign prostatic glands with preserved luminal and basal cell layers",
#         "normal prostate tissue with widely spaced glands and bland nuclei",
#         "benign prostate glands showing papillary infolding and regular contours",
#         "non-malignant prostatic epithelium with orderly glandular architecture",
#         "benign prostate acini containing luminal secretions",
#         "prostatic glands with corpora amylacea and no malignant features",
#         "benign prostatic hyperplasia with nodular glands and fibromuscular stroma",
#         "large well-circumscribed prostate glands lined by uniform epithelial cells",
#         "atrophic benign prostate glands without invasive carcinoma",
#         "histologically benign prostate tissue without adenocarcinoma",
#     ],
#     "stroma": [
#         "fibromuscular prostate stroma between glandular structures",
#         "collagenous prostatic stroma with spindle fibroblast nuclei",
#         "pink eosinophilic stromal collagen without epithelial glands",
#         "desmoplastic stroma surrounding invasive prostate carcinoma",
#         "loose fibrous prostate stroma with wavy collagen bundles",
#         "stromal connective tissue dominated by extracellular collagen",
#         "fibrovascular prostate stroma with sparse elongated stromal cells",
#         "fibrous stroma separating benign or malignant prostatic glands",
#     ],
#     "lymphocytes": [
#         "dense lymphocytic infiltrate of small dark round nuclei",
#         "aggregates of lymphocytes in H&E stained tissue",
#         "small monomorphic inflammatory lymphoid cells",
#         "blue-purple lymphocyte-rich inflammatory infiltrate",
#         "clusters of small mature lymphocytes with scant cytoplasm",
#         "lymphoid aggregate composed of many tiny dark nuclei",
#         "tumor infiltrating lymphocytes surrounding tissue structures",
#         "inflammatory area dominated by lymphocytes",
#     ],
#     "necrosis": [
#         "necrotic tissue with ghost cells and eosinophilic debris",
#         "acellular necrotic debris in H&E stained tissue",
#         "coagulative necrosis with loss of nuclear detail",
#         "dirty necrosis with granular cellular debris",
#         "dead tissue containing karyorrhectic nuclear fragments",
#         "pale necrotic region lacking viable cells",
#         "eosinophilic necrotic material with fragmented nuclei",
#         "area of tissue necrosis and cellular breakdown",
#     ],
#     "adipose tissue": [
#         "adipose tissue with large clear fat vacuoles",
#         "white fat cells with thin cytoplasmic rims",
#         "mature adipocytes forming empty round spaces",
#         "periprostatic fatty tissue dominated by mature adipocytes",
#         "clusters of large clear adipose cells in H&E",
#         "fat tissue with delicate septa between adipocytes",
#         "lipid vacuoles appearing as white empty spaces",
#         "benign adipose tissue with sparse nuclei",
#     ],
#     "tissue artifact": [
#         "tissue processing artifact with folds or tears",
#         "out-of-focus blurry histology artifact",
#         "crushed tissue artifact with distorted cells",
#         "section fold artifact in H&E stained tissue",
#         "air bubble or scanning artifact obscuring tissue",
#         "staining artifact with abnormal color or precipitate",
#         "damaged tissue edge with mechanical artifact",
#         "non-diagnostic tissue artifact rather than real histology",
#     ],
#     "blood vessel": [
#         "blood vessel lumen lined by endothelial cells",
#         "vascular structure containing red blood cells",
#         "artery or vein wall in H&E stained tissue",
#         "round vessel lumen filled with erythrocytes",
#         "endothelial-lined channel with intraluminal blood",
#         "small capillary or venule within stromal tissue",
#         "blood vessel with smooth wall and red cells",
#         "vascular lumen surrounded by connective tissue",
#     ],
#     "extracellular mucin": [
#         "extracellular mucin pools in H&E stained tissue",
#         "pale blue-gray mucin outside tumor cells",
#         "amorphous extracellular mucinous material",
#         "mucin lakes with floating epithelial tumor cells",
#         "gelatinous extracellular mucin separating tissue fragments",
#         "abundant mucin matrix with low cellularity",
#         "mucinous stroma containing pale extracellular mucin",
#         "acellular mucin pool in histopathology image",
#     ],
#     "nerve": [
#         "peripheral nerve bundle with wavy spindle nuclei",
#         "nerve fascicle in H&E stained tissue",
#         "neural tissue with elongated Schwann cell nuclei",
#         "pink nerve fibers arranged in a fascicular pattern",
#         "perineurial sheath surrounding a nerve bundle",
#         "wavy nerve fibers embedded in connective tissue",
#         "peripheral nerve within prostatic stroma",
#         "neural fascicle with parallel elongated nuclei",
#     ],
#     "hemorrhage": [
#         "hemorrhage with abundant extravasated red blood cells",
#         "blood-filled area outside vascular structures",
#         "red blood cells spilling through tissue",
#         "fresh hemorrhage in H&E stained section",
#         "dense collection of erythrocytes in tissue space",
#         "area dominated by red blood cells and blood clot",
#         "extravasated blood surrounding histologic tissue",
#         "hemorrhagic tissue with pools of erythrocytes",
#     ],
#     "smooth muscle": [
#         "prostatic smooth muscle bundles with elongated cigar-shaped nuclei",
#         "eosinophilic smooth muscle fibers in prostate fibromuscular stroma",
#         "pink spindle cell smooth muscle between prostate glands",
#         "muscular vessel wall composed of smooth muscle",
#         "interlacing bundles of prostatic smooth muscle cells",
#         "smooth muscle with uniform elongated nuclei",
#         "dense eosinophilic prostate muscle fibers without epithelial glands",
#         "fascicular smooth muscle tissue in prostate H&E section",
#     ],
#     "plasma cells": [
#         "plasma cells with eccentric nuclei and perinuclear hof",
#         "dense infiltrate of mature plasma cells",
#         "inflammatory plasma cells with clock-face chromatin",
#         "oval plasma cells with basophilic cytoplasm",
#         "clusters of antibody-producing plasma cells",
#         "plasmacytic infiltrate in H&E stained tissue",
#         "eccentric nuclei and abundant cytoplasm in plasma cells",
#         "chronic inflammation rich in plasma cells",
#     ],
# }

## 作者github中的prompt
# PROMPTS_BY_CLASS = {
#     "malignant tissue": [
#         "an H&E stained image of malignant tissue.",
#         "a photomicrograph showing malignant tissue.",
#         "tissue section showing malignant tissue.",
#         "area dominated by malignant tissue.",
#         "patch with abundant malignant tissue.",
#         "region with evidence of malignant tissue.",
#         "an example of malignant tissue.",
#         "this is malignant tissue.",
#         "presence of malignant tissue.",
#         "malignant tissue is present.",
#         "an image of malignant tissue.",
#         "a histopathological photograph of malignant tissue.",
#         "a histopathological image of malignant tissue.",
#     ],

#     "benign tissue": [
#         "an H&E stained image of benign tissue.",
#         "a photomicrograph showing benign tissue.",
#         "tissue section showing benign tissue.",
#         "area dominated by benign tissue.",
#         "patch with abundant benign tissue.",
#         "region with evidence of benign tissue.",
#         "an example of benign tissue.",
#         "this is benign tissue.",
#         "presence of benign tissue.",
#         "benign tissue is present.",
#         "an image of benign tissue.",
#         "a histopathological photograph of benign tissue.",
#         "a histopathological image of benign tissue.",
#     ],

#     "stroma": [
#         "an H&E stained image of stroma.",
#         "a photomicrograph showing stroma.",
#         "tissue section showing stroma.",
#         "area dominated by stroma.",
#         "patch with abundant stroma.",
#         "region with evidence of stroma.",
#         "an example of stroma.",
#         "this is stroma.",
#         "presence of stroma.",
#         "stroma is present.",
#         "an image of stroma.",
#         "a histopathological photograph of stroma.",
#         "a histopathological image of stroma.",
#     ],

#     "lymphocytes": [
#         "an H&E stained image of lymphocytes.",
#         "a photomicrograph showing lymphocytes.",
#         "tissue section showing lymphocytes.",
#         "area dominated by lymphocytes.",
#         "patch with abundant lymphocytes.",
#         "region with evidence of lymphocytes.",
#         "an example of lymphocytes.",
#         "this is lymphocytes.",
#         "presence of lymphocytes.",
#         "lymphocytes are present.",
#         "an image of lymphocytes.",
#         "a histopathological photograph of lymphocytes.",
#         "a histopathological image of lymphocytes.",
#     ],

#     "necrosis": [
#         "an H&E stained image of necrosis.",
#         "a photomicrograph showing necrosis.",
#         "tissue section showing necrosis.",
#         "area dominated by necrosis.",
#         "patch with abundant necrosis.",
#         "region with evidence of necrosis.",
#         "an example of necrosis.",
#         "this is necrosis.",
#         "presence of necrosis.",
#         "necrosis is present.",
#         "an image of necrosis.",
#         "a histopathological photograph of necrosis.",
#         "a histopathological image of necrosis.",
#     ],

#     "adipose tissue": [
#         "an H&E stained image of adipose tissue.",
#         "a photomicrograph showing adipose tissue.",
#         "tissue section showing adipose tissue.",
#         "area dominated by adipose tissue.",
#         "patch with abundant adipose tissue.",
#         "region with evidence of adipose tissue.",
#         "an example of adipose tissue.",
#         "this is adipose tissue.",
#         "presence of adipose tissue.",
#         "adipose tissue is present.",
#         "an image of adipose tissue.",
#         "a histopathological photograph of adipose tissue.",
#         "a histopathological image of adipose tissue.",
#     ],

#     "tissue artifact": [
#         "an H&E stained image of tissue artifact.",
#         "a photomicrograph showing tissue artifact.",
#         "tissue section showing tissue artifact.",
#         "area dominated by tissue artifact.",
#         "patch with abundant tissue artifact.",
#         "region with evidence of tissue artifact.",
#         "an example of tissue artifact.",
#         "this is tissue artifact.",
#         "presence of tissue artifact.",
#         "tissue artifact is present.",
#         "an image of tissue artifact.",
#         "a histopathological photograph of tissue artifact.",
#         "a histopathological image of tissue artifact.",
#     ],

#     "blood vessel": [
#         "an H&E stained image of blood vessel.",
#         "a photomicrograph showing blood vessel.",
#         "tissue section showing blood vessel.",
#         "area dominated by blood vessel.",
#         "patch with abundant blood vessel.",
#         "region with evidence of blood vessel.",
#         "an example of blood vessel.",
#         "this is blood vessel.",
#         "presence of blood vessel.",
#         "blood vessel is present.",
#         "an image of blood vessel.",
#         "a histopathological photograph of blood vessel.",
#         "a histopathological image of blood vessel.",
#     ],

#     "extracellular mucin": [
#         "an H&E stained image of extracellular mucin.",
#         "a photomicrograph showing extracellular mucin.",
#         "tissue section showing extracellular mucin.",
#         "area dominated by extracellular mucin.",
#         "patch with abundant extracellular mucin.",
#         "region with evidence of extracellular mucin.",
#         "an example of extracellular mucin.",
#         "this is extracellular mucin.",
#         "presence of extracellular mucin.",
#         "extracellular mucin is present.",
#         "an image of extracellular mucin.",
#         "a histopathological photograph of extracellular mucin.",
#         "a histopathological image of extracellular mucin.",
#     ],

#     "nerve": [
#         "an H&E stained image of nerve.",
#         "a photomicrograph showing nerve.",
#         "tissue section showing nerve.",
#         "area dominated by nerve.",
#         "patch with abundant nerve.",
#         "region with evidence of nerve.",
#         "an example of nerve.",
#         "this is nerve.",
#         "presence of nerve.",
#         "nerve is present.",
#         "an image of nerve.",
#         "a histopathological photograph of nerve.",
#         "a histopathological image of nerve.",
#     ],

#     "hemorrhage": [
#         "an H&E stained image of hemorrhage.",
#         "a photomicrograph showing hemorrhage.",
#         "tissue section showing hemorrhage.",
#         "area dominated by hemorrhage.",
#         "patch with abundant hemorrhage.",
#         "region with evidence of hemorrhage.",
#         "an example of hemorrhage.",
#         "this is hemorrhage.",
#         "presence of hemorrhage.",
#         "hemorrhage is present.",
#         "an image of hemorrhage.",
#         "a histopathological photograph of hemorrhage.",
#         "a histopathological image of hemorrhage.",
#     ],

#     "smooth muscle": [
#         "an H&E stained image of smooth muscle.",
#         "a photomicrograph showing smooth muscle.",
#         "tissue section showing smooth muscle.",
#         "area dominated by smooth muscle.",
#         "patch with abundant smooth muscle.",
#         "region with evidence of smooth muscle.",
#         "an example of smooth muscle.",
#         "this is smooth muscle.",
#         "presence of smooth muscle.",
#         "smooth muscle is present.",
#         "an image of smooth muscle.",
#         "a histopathological photograph of smooth muscle.",
#         "a histopathological image of smooth muscle.",
#     ],

#     "plasma cells": [
#         "an H&E stained image of plasma cells.",
#         "a photomicrograph showing plasma cells.",
#         "tissue section showing plasma cells.",
#         "area dominated by plasma cells.",
#         "patch with abundant plasma cells.",
#         "region with evidence of plasma cells.",
#         "an example of plasma cells.",
#         "this is plasma cells.",
#         "presence of plasma cells.",
#         "plasma cells are present.",
#         "an image of plasma cells.",
#         "a histopathological photograph of plasma cells.",
#         "a histopathological image of plasma cells.",
#     ],
# }



def build_prompts() -> dict[str, list[str]]:
    missing = [label for label in CLASSES if label not in PROMPTS_BY_CLASS]
    if missing:
        raise ValueError(f"Missing prompts for classes: {missing}")
    return {label: PROMPTS_BY_CLASS[label] for label in CLASSES}


def load_conch_model(checkpoint_path: str, device: torch.device):
    print(f"Loading CONCH model from: {checkpoint_path}")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    model.eval()
    model = model.to(device)
    print("Model loaded")
    return model


def encode_texts(model, queries: list[str], device: torch.device) -> np.ndarray:
    tokenizer = get_tokenizer()
    tokenized = tokenize(tokenizer, queries).to(device)
    with torch.inference_mode():
        feats = model.encode_text(tokenized, normalize=True)
    feats = feats.cpu().numpy().astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats


def encode_class_prompts(model, device: torch.device):
    prompts_by_class = build_prompts()
    class_features = []
    prompt_features = []
    prompt_labels = []

    for class_idx, label in enumerate(CLASSES):
        prompts = prompts_by_class[label]
        print(f"\nEncoding class: {label} ({len(prompts)} prompts)")
        for i, prompt in enumerate(prompts):
            print(f"  [{i}] {prompt}")

        feats = encode_texts(model, prompts, device)
        prototype = feats.mean(axis=0)
        prototype /= np.linalg.norm(prototype) + 1e-12

        class_features.append(prototype.astype(np.float32))
        prompt_features.append(feats)
        prompt_labels.extend([class_idx] * len(prompts))

        print(f"  prompt_features: {feats.shape}")
        print(f"  class prototype norm: {np.linalg.norm(prototype):.4f}")

    return (
        np.stack(class_features, axis=0),
        np.vstack(prompt_features),
        np.array(prompt_labels, dtype=np.int64),
        prompts_by_class,
    )


def save_outputs(
    output_dir: str,
    class_features: np.ndarray,
    prompt_features: np.ndarray,
    prompt_labels: np.ndarray,
    prompts_by_class: dict[str, list[str]],
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "class_features.npy"), class_features)
    np.save(os.path.join(output_dir, "prompt_features.npy"), prompt_features)
    np.save(os.path.join(output_dir, "prompt_labels.npy"), prompt_labels)

    with open(os.path.join(output_dir, "classes.txt"), "w") as handle:
        for label in CLASSES:
            handle.write(label + "\n")

    with open(os.path.join(output_dir, "label_colors.json"), "w") as handle:
        json.dump({k: list(v) for k, v in LABEL_TO_COLOR.items()}, handle, indent=2)

    with open(os.path.join(output_dir, "prompts.json"), "w") as handle:
        json.dump(prompts_by_class, handle, indent=2)

    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(
            {
                "prompt_version": PROMPT_VERSION,
                "num_classes": len(CLASSES),
                "num_prompts_per_class": {label: len(prompts_by_class[label]) for label in CLASSES},
            },
            handle,
            indent=2,
        )

    with open(os.path.join(output_dir, "queries.txt"), "w") as handle:
        for class_idx, label in enumerate(CLASSES):
            handle.write(f"=== Class {class_idx}: {label} ===\n")
            for prompt_idx, prompt in enumerate(prompts_by_class[label]):
                handle.write(f"[{prompt_idx}] {prompt}\n")
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Encode multi-class CONCH text prompts")
    parser.add_argument("--checkpoint_path", required=True, help="Path to CONCH pytorch_model.bin")
    parser.add_argument("--output_dir", default="./text_features", help="Directory to save features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {CLASSES}")

    model = load_conch_model(args.checkpoint_path, device)
    class_features, prompt_features, prompt_labels, prompts_by_class = encode_class_prompts(model, device)
    save_outputs(args.output_dir, class_features, prompt_features, prompt_labels, prompts_by_class)

    print("\nSaved:")
    print(f"  {os.path.join(args.output_dir, 'class_features.npy')} {class_features.shape}")
    print(f"  {os.path.join(args.output_dir, 'prompt_features.npy')} {prompt_features.shape}")
    print(f"  {os.path.join(args.output_dir, 'prompt_labels.npy')} {prompt_labels.shape}")
    print(f"  {os.path.join(args.output_dir, 'classes.txt')}")
    print(f"  {os.path.join(args.output_dir, 'prompts.json')}")
    print("\nDone")


if __name__ == "__main__":
    main()
