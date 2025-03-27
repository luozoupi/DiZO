## DiZO: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning 
Implementation of the divergence-driven zeroth-order optimization ([PDF](https://arxiv.org/pdf/2502.03304)).

There are still some bugs when applying First-order for projection ($\gamma$) searching, which may cause the memory to keep increasing during training, and we will fix it as soon as possible. You can reproduce the results in the paper by running the following.
```bash
# do not involve $\gamma$
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 bash mezo.sh
# use zo for $\gamma$ searching
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 ENHANCED=zo bash mezo.sh
# use fo for $\gamma$ searching
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 ENHANCED=fo bash mezo.sh
```

## Some tips in training:
* `ZO` is sensitive to hyperparameter settings (including seeds).
* For `DiZO`, in easier datasets (e.g. SST-2), you can apply more aggressive projection, i.e., larger projection scalar or more frequent projection learning, and vice versa.

## How to add DiZO to my own code?

Our implementation of DiZO is based on [MeZO](https://github.com/princeton-nlp/MeZO). For the adding parts, please refer to `trainer.py` for details (to see where we added, search `DiZO added`).
