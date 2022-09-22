# MSc-Thesis-Cryptocurrency-Price-Forecasting-Using-BERT-Based-Sentiment

Cryptocurrency Return Prediction Using Investor Sentiment Extracted by BERT-Based Classifiers From News Articles, Reddit Posts and Tweets

Master's thesis project for the program of M.Sc. Economics and Management Science at Humboldt University of Berlin

--by Duygu Ider   https://www.linkedin.com/in/duyguider/

Please find the paper here: https://arxiv.org/abs/2204.05781

Outline of the project and what each script/notebook does:

PART 1: BERT-Based Sentiment Classification
1. _price_data_scrape.ipynb_ - Scrape price data for Bitcoin and Ethereum
2. _news_scraper_final.ipynb_, _reddit_scraper_final.ipynb_, _twitter_scraper_final.py_ - Scrape news, Reddit and Tweets data
3. _weak_labels_approach.py_ - Use Financhial Phrasebank data (Malo et. al, 2014) to label it with pseudo-labels predicted by BART zero-shot classifier, fit a BERT-based classifier, evaluate model performance in the case of weak labels
4. _combine_text_data_zsc_finbert.py_ - Combine the price and text data to a single dataset, predict sentiment using zero-shot classifier (BART) and FinBERT to assign weak labels
5. _bert_crypto_hyperparam_optimal_and_zsc.ipynb_ - Perform grid search hyperparameter optimization to the process of fine-tuning BERT-based classifiers. The implemented models are BERT-Unfrozen, BERT-Frozen and BERT-Context

PART 2: Return Prediction and Trading Simulation
1. _data_prep_for_financial_models.py_ - Prepare the combined dataset as an input for the financial models. Add price, macroeconomic, blockchain features and weekday dummies
2. _return_prediction_trading_simulation.ipynb_ - Load data, add technical analsis features to the dataset, lag defined features by a certain lag amount, plot some intermediate outputs, perform elimination by variance inflation factor to analyze sentiment feature contribution, fit all cryptocurrency return predictors using Bayesian hyperparameter optimization, perform trading simulation over multiple test periods, create a clearly defined output table of all prediction results
3. _return_prediction_trading_simulation_(rnn_added_pipeline_implemented).ipynb_ - RNN and LSTM added as financial forecasting models, compared to the previous script
