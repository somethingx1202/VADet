package main.org.backingdata.twitter.crawler.util;

import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;

public class CredentialObject {

	// Authentication
	private String consumerKey = "";
	private String consumerSecret = "";
	private String token = "";
	private String tokenSecret = "";

	private boolean isValid = false;
	private String isValidExplanation = "Credentials not checked for validity";

	// Setters and getters
	public String getConsumerKey() {
		return consumerKey;
	}

	public void setConsumerKey(String consumerKey) {
		this.consumerKey = consumerKey;
	}

	public String getConsumerSecret() {
		return consumerSecret;
	}

	public void setConsumerSecret(String consumerSecret) {
		this.consumerSecret = consumerSecret;
	}

	public String getToken() {
		return token;
	}

	public void setToken(String token) {
		this.token = token;
	}

	public String getTokenSecret() {
		return tokenSecret;
	}

	public void setTokenSecret(String tokenSecret) {
		this.tokenSecret = tokenSecret;
	}

	public boolean isValid() {
		return isValid;
	}

	public void setValid(boolean isValid) {
		this.isValid = isValid;
	}

	public String getIsValidExplanation() {
		return isValidExplanation;
	}

	public void setIsValidExplanation(String isValidExplanation) {
		this.isValidExplanation = isValidExplanation;
	}
	
	


	@Override
	public String toString() {
		return "CredentialObject [consumerKey=" + ((consumerKey != null) ? consumerKey : "NULL") + 
				", consumerSecret=" + ((consumerSecret != null) ? consumerSecret : "NULL") + 
				", token=" + ((token != null) ? token : "NULL") + 
				", tokenSecret=" + ((tokenSecret != null) ? tokenSecret : "NULL") + 
				", isValid=" + isValid + 
				", isValidExplanation="	+ ((isValidExplanation != null) ? isValidExplanation : "NULL") + "]";
	}

	public static String validateCredentials(CredentialObject credObjToValidate) {

		try {
			ConfigurationBuilder cb = new ConfigurationBuilder();
			cb.setDebugEnabled(true).setJSONStoreEnabled(true);

			TwitterFactory tf = new TwitterFactory(cb.build());

			Twitter twitter = tf.getInstance();
			AccessToken accessToken = new AccessToken(credObjToValidate.getToken(), credObjToValidate.getTokenSecret());
			twitter.setOAuthConsumer(credObjToValidate.getConsumerKey(), credObjToValidate.getConsumerSecret());
			twitter.setOAuthAccessToken(accessToken);
			
			Status retrivedTweet = twitter.showStatus(Long.valueOf("823144056373018624"));
			
			if(retrivedTweet != null && retrivedTweet.getText() != null && !retrivedTweet.getText().trim().equals("")) {
				return null;
			}
			else {
				return "Invalid credential object.";
			}
			
		} 
		catch(Exception e) {
			return "Invalid credential object > " + ((e.getMessage() != null) ? e.getMessage() : "NULL");
		}
	}

}
