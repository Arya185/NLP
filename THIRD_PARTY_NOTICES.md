# Third-Party Notices and Attribution

This file records upstream sources used by this project and the licensing questions that are still open.
It is **not legal advice and does not claim that any licensing question has been resolved.** Facts below were
checked against public pages on 2026-10-28; anything not verified is marked **REVIEW**.

## Scope of the project's own licence

The project's original source code is intended to be released under the MIT License. **A `LICENSE` file has not
been added yet** (the copyright holder still has to be confirmed), so no licence is granted for the project's code
until it is. When added, the MIT License will apply **only to the project's original code**. It does **not** grant
any rights to third-party datasets, third-party model artifacts, or anything derived from them, and it does not
replace any upstream licence.

## 1. `google/muril-base-cased` (base model)

- **What it is:** the pretrained multilingual model that `train_muril.py` fine-tunes.
- **Reported licence:** Apache-2.0, as declared in the model's Hugging Face metadata (checked 2026-10-28).
- **REVIEW:** the model repository's file list contains no separate `LICENSE` file (it holds `README.md`,
  `config.json`, `tokenizer_config.json`, `vocab.txt` and the weight files), so the Apache-2.0 declaration has
  **not been verified against an upstream licence text**. This project does not reproduce any licence text. Before
  publishing derived files, confirm the licence terms and any notice obligations that apply.
- **Derived artifacts:** the fine-tuned weights (`model.safetensors`) and the tokenizer and configuration files
  distributed with them (`tokenizer.json`, `tokenizer_config.json`, `config.json`) are derived from this base model.
  The project's MIT licence does not replace the upstream licence for these files, and the files remain subject to
  whatever terms apply to the base model.
- **Citation requested by the model card:** Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., Dey, A., Gopalan, B.,
  Margam, D. K., Aggarwal, P., Nagipogu, R. T., Dave, S., Gupta, S., Gali, S. C. B., Subramanian, V., and
  Talukdar, P. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730 [cs.CL].

## 2. Davidson et al. hate-speech and offensive-language dataset

- **What it is:** the English training data used by `train_baseline.py` and `train_muril.py`
  (`labeled_data.csv`, classes 0 hate speech, 1 offensive language, 2 neither).
- **Citation** (verified against the authors' repository README and the AAAI proceedings page):
  Davidson, T., Warmsley, D., Macy, M., and Weber, I. (2017). Automated Hate Speech Detection and the Problem of
  Offensive Language. *Proceedings of the International AAAI Conference on Web and Social Media*, 11(1), 512-515.
  https://doi.org/10.1609/icwsm.v11i1.14955
- **Authors' repository licence:** MIT (License file reads "Copyright (c) 2017 Tom Davidson"). That repository's
  README and data folder contain no statement about the terms of the tweet text itself.
- **Raw data is not included** in this project's source repository, and users must independently verify that
  they may access and use it. **Do not redistribute it.**
- **REVIEW (open):** the dataset consists of tweets (third-party content). X's developer policy page, read on
  2026-10-28, states that redistribution of X Content to third parties is restricted to IDs, with limited
  exceptions. Whether and how that applies to this dataset has **not** been established. The MIT licence of the
  authors' repository does not settle it.

## 3. Artifacts derived from the tweet data (REVIEW)

- **SVM model (`models/svm_pipeline.joblib`):** its TF-IDF vocabulary and weights were learned from the dataset
  above. It is **excluded from the source repository** pending review, and users generate it locally. It contains
  no raw tweets, but whether redistributing it is acceptable has **not** been established.
- **Fine-tuned MuRIL weights:** trained on the same data. Whether and how they may be redistributed (for example
  as a GitHub Release asset) has **not** been established, in addition to the base-model question in section 1.

## 4. Other material (REVIEW)

- **Hinglish seed sentences** (`src/data.py`, 15 sentences): described in the project documentation as hand-written
  for this project. Their provenance should be confirmed by the maintainer. They contain hateful and offensive
  wording by design.
- **Early exploratory Colab notebooks** (`colab_training_pipeline.ipynb`, `Hate_Speech_Detection_Colab.ipynb`):
  the origin of their code is not documented here and should be confirmed before publication.
- **Python dependencies** (Streamlit, scikit-learn, pandas, numpy, joblib, emoji, pytest, PyTorch, Transformers
  and their dependencies) are installed with `pip` and are not redistributed by this repository. Their licences
  have not been reviewed here.
