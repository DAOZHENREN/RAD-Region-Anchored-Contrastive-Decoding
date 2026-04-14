# Model checkpoints
CHEXBERT_PATH = "/app/Models/hub/models--StanfordAIMI--RRG_scorers/snapshots/6646433b3ad83a10f6e141db76d0ece44312b236/chexbert.pth"
RADGRAPH_PATH = "/app/Models/radgraph-extracting-clinical-entities-and-relations-from-radiology-reports-1.0.0/models/model_checkpoint/model.tar.gz"

# Report paths
GT_REPORTS = "reports/gt_reports.csv"
PREDICTED_REPORTS = "reports/predicted_reports.csv"
OUT_FILE = "report_scores.csv"

# Whether to use inverse document frequency (idf) for BERTScore
USE_IDF = False
