# VADet
VAD contains the source code and the dataset for vaccine attitude detection.

### Vaccine Attitude Dataset

The annotations are given in the form of `ID,stance,aspect_span_start:aspect_span_end,opinion_span_start:opinion_span_end,aspect_catetegory` <br />
in the `Datasets_Raw` folder. <br />

To obtain tweet text,
1.  `cd twitter_get_text_by_id_twitter4j`
2.  Open `./settings/crawler.properties` and setup your `consumerKey, consumerSecret, access token and access token secret`.
    1. For the acquisition of `consumerKey, consumerSecret, access token and access token secret`, please refer to https://developer.twitter.com/en/docs/developer-portal/overview. The Standard v1.1 is sufficient.

