# Human vs Model Experiment

## Methodology

Participants were asked to fill out a Google Form given these instructions: 

You will be shown a series of messages (emails or SMS). For each message:

A. Read the message carefully.
B. Use the slider (0–10) to indicate how confident you are that the message is spam: 
0 = You are EXTREMELY confident the message is legitimate (Ham)
10 = You are EXTREMELY confident the message is spam (Illegitimate)
C. Take your time, there are no right or wrong answers, but answer honestly based on your intuition.

Then participants read 30 messages and ranked them on a scale from 1 to 10 based on how confident they are that the messages are illegitimate. In other words, ranking a message a 10 means they are VERY confident that the message is a scam. 

## Data

See the subset used in the human test in `data/human_test_set.csv`. The data in this sample test set was randomly chosen to remove personal bias from the selection. Participants were not told but there were an equal amount of spam and ham messages.

Link to form: https://forms.gle/fFMn1HoJpQfZ7gH1A

## Results

The Google Form data was saved to `data/human_responses.csv` for analysis. 20 people filled out the form. 

### Model Performance vs Human Alignment

While both models achieved high accuracy on binary classification (>97%), their alignment with human judgment showed significant discrepancies:

**Correlation with Human Mean Scores:**
- Naive Bayes: 0.4711
- Logistic Regression: 0.4873

**Mean Absolute Error (0-10 scale):**
- Naive Bayes: 4.45
- Logistic Regression:  4.37

Both models showed only moderate correlation (~0.47-0.49) with human judgment, indicating that while they accurately learned the training labels, those labels don't always align with human intuition about what constitutes spam.

### Biggest Disagreements

The top 5 disagreements between models and humans reveal systematic differences:

| Question | Human Mean | NB Score | LR Score | Text Preview |
|----------|-----------|----------|----------|--------------|
| 4 | 9.2 | 0.0 | 1.3 | "Will u meet ur dream partner soon? Is ur career off 2 a flyng start? ..." |
| 23 | 8.2 | 0.0 | 0.1 | "Today is song dedicated day..  Which song will u dedicate for me? ..." |
| 9 | 7.9 | 0.0 | 0.6 | "How about getting in touch with folks waiting for company? Just txt back..." |
| 17 | 7.7 | 0.0 | 0.1 | "Go until jurong point, crazy..  Available only in bugis n great world..." |
| 26 | 7.2 | 0.0 | 0.2 | "Hello handsome!  Are you finding that job?  Not being lazy? ..." |

### Key Findings

1. **Models are not overfitting**: Cross-validation results show stable performance (NB: 97. 1% ± 0.76%, LR: 97.8% ± 0.55%), confirming the models generalize well.

2. **Category differences**: Humans appear to classify the following as spam more readily than the training labels: 
   - Messages requiring action ("txt back", "send to friends")
   - Informal/abbreviated language combined with promotional content
   - Unsolicited services even if not explicitly commercial

3. **Model confidence and human hesitation**:  In disagreement cases, models are highly confident (scores near 0), suggesting these patterns are consistently labeled as "ham" in the training set, not edge cases. In contrast, humans were less likely to answer on the extremes.

### Implications

This experiment reveals that binary spam classification accuracy doesn't fully capture alignment with human judgment. The ~50% correlation suggests that subjective spam perception differs significantly from the SMS Spam Collection dataset labels, particularly for: 
- Social chain messages
- Opt-in promotional content
- Informal service offerings

The detailed comparison data is available in `results/human_model_comparison.csv` for further analysis.
