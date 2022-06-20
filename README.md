# VADet
This repository contains the source code and the dataset for vaccine attitude detection.

## Vaccine Attitude Dataset

The annotations are given in the form of `ID,stance,aspect_span_start:aspect_span_end,opinion_span_start:opinion_span_end,aspect_catetegory` <br />
in the `Datasets_Raw` folder. <br />

To obtain tweet text,
1.  `cd twitter_get_text_by_id_twitter4j`
2.  Open `./settings/crawler.properties` and setup your `consumerKey, consumerSecret, access token and access token secret`.
    1. For the acquisition of `consumerKey, consumerSecret, access token and access token secret`, please refer to https://developer.twitter.com/en/docs/developer-portal/overview. The Standard v1.1 is sufficient.
3.  run twitter_get_text_by_id_twitter4j by either `java -jar twitter_vac_opi_cwl_by_id.jar ./settings/crawler.properties` or `javac -cp "./*" ./src/main/org/backingdata/twitter/crawler/rest/TwitterRESTTweetIDlistCrawler.java` The tweets are stored in `./saves` in json format.

## VAD unsupervised training
`cd VADMlmFineTuning`
VADtransformer is firstly trained unsupervised and saved to `../datasets/mlm-vad`. To perform unsupervised training,
1.  Replace tweetIDs in `UnannotatedTwitterID_training.csv` and `UnannotatedTwitterID_testing.csv` with obtained tweet text.
2.  Put the tweet text file in `../datasets`. The format is the same as `vad_train_finetune.txt`.
3.  `cd src` and run `train_vad_albert_vae.py`

## VAD supervised training
`cd VADStanceAndTextspanPrediction`

For your convenience the pre-trained TopicDrivenMaskedLM is ready-to-use in the corresponding folder of the repository.
