# ComplaintPrompt
Distributed and parallel Naive-Bayes multi-class text classification for large-scale
complaint analysis.

# To run on your local virtual environment:

python ComplaintClassificator.py [PATH_TO_ConsumerComplaint.csv] [PATH_TO_AmericanStatesAbb.json]

# To run on your cluster set-up:

python ComplaintClassificator.py [PATH_TO_ConsumerComplaint.csv] [PATH_TO_AmericanStatesAbb.json] [AWS_ACCESS_KEY_ID] [AWS_SECRET_KEY_ACCESS]

PATH_TO_ConsumerComplaint.csv - can be read from S3. Files > 5GB are also handles as implementation of the s3a filesystem is used to retrieved them.

