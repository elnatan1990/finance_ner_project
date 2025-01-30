Install process:

pip install torch transformers datasets seqeval numpy accelerate sacremoses tqdm

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

===

LABELS:

What we want to detect?

self.labels = ["O", "name", "company", "street_address"]  # Changed to lowercase

===

PREDICT:

1. NER train & test on 'company', 'name', 'street_address' data:

C:\Users\elnat\miniconda3\envs\pythonProject\python.exe C:\Users\elnat\py\NER\pythonProject\NER4.pyLoading model...

Model loaded from ./ner_model


Predicting entities in text:


A. Input: John Smith from Apple Inc sent a letter to 123 Main Street, New York.

Entities found:- JohnSmith: name- AppleInc: company- 123MainStreet,NewYork: street_address


B. Input: Microsoft Corporation's CEO Satya Nadella visited the office at 456 Tech Boulevard.

Entities found:- MicrosoftCorporation: company- SatyaNadella: name- 4456TechBoulevard: street_address


C. Input: Sarah Johnson works at Deutsche Bank in Frankfurt.

Entities found:- SarahJohnson: name- DeutscheBank: company

===

TRAIN PROCESS:

2. DEBUG FROM TRAIN:

Debug: Detailed metrics per label:

Step: 536/665 (80.6%)
Time elapsed: 00:34:10

Evaluation Metrics:
Macro F1: 0.8288

*** New best macro F1: 0.8288 ***
{'eval_loss': 0.03269880637526512, 'eval_macro_precision': 0.798123080691561, 'eval_macro_recall': 0.8652743408814824, 'eval_macro_f1': 0.8288248610926532, 'eval_runtime': 5.9513, 'eval_samples_per_second': 71.918, 'eval_steps_per_second': 4.537, 'epoch': 4.0}
                                                 
 81%|████████  | 536/665 [34:10<06:15,  2.91s/it]
100%|██████████| 27/27 [00:05<00:00,  7.36it/s]
 83%|████████▎ | 550/665 [35:07<07:15,  3.79s/it]
Step: 550/665 (82.7%)
Time elapsed: 00:35:07
Training loss: 0.0302
{'loss': 0.0302, 'grad_norm': 0.41070857644081116, 'learning_rate': 1.973244147157191e-06, 'epoch': 4.1}
 90%|█████████ | 600/665 [38:16<04:04,  3.77s/it]
Step: 600/665 (90.2%)
Time elapsed: 00:38:16
Training loss: 0.0273
{'loss': 0.0273, 'grad_norm': 0.6418583989143372, 'learning_rate': 1.137123745819398e-06, 'epoch': 4.48}
 98%|█████████▊| 650/665 [41:24<00:56,  3.77s/it]
Step: 650/665 (97.7%)
Time elapsed: 00:41:24
Training loss: 0.0286
{'loss': 0.0286, 'grad_norm': 0.3538336753845215, 'learning_rate': 3.010033444816054e-07, 'epoch': 4.85}
100%|██████████| 665/665 [42:20<00:00,  3.76s/it]
  0%|          | 0/27 [00:00<?, ?it/s]
  7%|▋         | 2/27 [00:00<00:01, 14.31it/s]
 15%|█▍        | 4/27 [00:00<00:02,  9.09it/s]
 22%|██▏       | 6/27 [00:00<00:02,  8.15it/s]
 26%|██▌       | 7/27 [00:00<00:02,  7.95it/s]
 30%|██▉       | 8/27 [00:00<00:02,  7.79it/s]
 33%|███▎      | 9/27 [00:01<00:02,  7.65it/s]
 37%|███▋      | 10/27 [00:01<00:02,  7.58it/s]
 41%|████      | 11/27 [00:01<00:02,  7.51it/s]
 44%|████▍     | 12/27 [00:01<00:02,  7.47it/s]
 48%|████▊     | 13/27 [00:01<00:01,  7.40it/s]
 52%|█████▏    | 14/27 [00:01<00:01,  7.42it/s]
 56%|█████▌    | 15/27 [00:01<00:01,  7.37it/s]
 59%|█████▉    | 16/27 [00:02<00:01,  7.43it/s]
 63%|██████▎   | 17/27 [00:02<00:01,  7.42it/s]
 67%|██████▋   | 18/27 [00:02<00:01,  7.40it/s]
 70%|███████   | 19/27 [00:02<00:01,  7.42it/s]
 74%|███████▍  | 20/27 [00:02<00:00,  7.36it/s]
 78%|███████▊  | 21/27 [00:02<00:00,  7.28it/s]
 81%|████████▏ | 22/27 [00:02<00:00,  7.33it/s]
 85%|████████▌ | 23/27 [00:03<00:00,  7.35it/s]
 89%|████████▉ | 24/27 [00:03<00:00,  7.36it/s]
 93%|█████████▎| 25/27 [00:03<00:00,  7.44it/s]
 96%|█████████▋| 26/27 [00:03<00:00,  7.34it/s]
Debug: Raw predictions shape: (428, 512)
Debug: Raw labels shape: (428, 512)

Debug: Sample predictions vs actual (first 3 sequences):

Sequence 1:
Predicted: O               Actual: street_address
Predicted: O               Actual: street_address
Predicted: O               Actual: street_address
Predicted: O               Actual: street_address
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O

Sequence 2:
Predicted: company         Actual: O
Predicted: company         Actual: O
Predicted: company         Actual: O
Predicted: company         Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O
Predicted: street_address  Actual: O

Sequence 3:
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company
Predicted: company         Actual: company

Debug: Detailed metrics per label:

Step: 665/665 (100.0%)
Time elapsed: 00:42:33

Evaluation Metrics:
Macro F1: 0.8294

*** New best macro F1: 0.8294 ***
{'eval_loss': 0.032171547412872314, 'eval_macro_precision': 0.7993433318308735, 'eval_macro_recall': 0.8642641903358889, 'eval_macro_f1': 0.829396988752022, 'eval_runtime': 6.3242, 'eval_samples_per_second': 67.676, 'eval_steps_per_second': 4.269, 'epoch': 4.97}
                                                 
100%|██████████| 665/665 [42:33<00:00,  3.76s/it]
100%|██████████| 27/27 [00:06<00:00,  7.34it/s]
100%|██████████| 665/665 [42:41<00:00,  3.76s/it]
Step: 665/665 (100.0%)
Time elapsed: 00:42:41
{'train_runtime': 2561.9972, 'train_samples_per_second': 8.318, 'train_steps_per_second': 0.26, 'train_loss': 0.10776972170162918, 'epoch': 4.97}
100%|██████████| 665/665 [42:42<00:00,  3.85s/it]

Model and configuration saved to ./ner_model

Process finished with exit code 0
