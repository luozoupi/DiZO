There are still some bugs when applying First-order for projection ($\gamma$) searching, which may cause the memory to keep increasing during training, and we will fix it as soon as possible. You can reproduce the results in the paper by running the following.
```bash
# do not involve $\gamma$
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 bash mezo.sh
# use zo for $\gamma$ searching
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 ENHANCED=zo bash mezo.sh
# use fo for $\gamma$ searching
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-6 EPS=1e-3 STEPS=4000 ENHANCED=fo bash mezo.sh
```
