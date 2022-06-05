package main.org.backingdata.twitter.crawler;

import java.util.Iterator;
import java.util.logging.Logger;

import twitter4j.Paging;
import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.ResponseList;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;

/**
 * This is an example class that shows how to exploit the Twitter4J java library to interact with Twitter
 * 
 * Twitter4j: http://twitter4j.org/en/index.html
 * Download (version 4.0.1):http://twitter4j.org/archive/twitter4j-4.0.1.zip
 * JavaDoc: http://twitter4j.org/javadoc/index.html
 * Example code of Twitter4j: http://twitter4j.org/en/code-examples.html
 * 
 * @author Francesco Ronzano
 *
 */
public class Twitter4jExampleREST {

	private static Logger logger = Logger.getLogger(Twitter4jExampleREST.class.getName());
	
	public static void main(String[] args) {
		
		// 1) Instantiate a Twitter Factory
		ConfigurationBuilder cb = new ConfigurationBuilder();
		cb.setDebugEnabled(true).setJSONStoreEnabled(true);
		TwitterFactory tf = new TwitterFactory(cb.build());
		
		// 2) Instantiate a new Twitter client
		// Go to https://dev.twitter.com/ to register a new Twitter App and get credentials
		Twitter twitter = tf.getInstance();
		AccessToken accessToken = new AccessToken("PUT_YOUR_ACCESS_TOKEN", "PUT_YOUR_ACCESS_TOKEN_SECRET");
		twitter.setOAuthConsumer("PUT_YOUR_API_KEY", "PUT_YOUR_API_SECRET");
		twitter.setOAuthAccessToken(accessToken);
		
		// Task1: search for all the tweets with the keywords: football world cup
	    // Reference JavaDoc http://twitter4j.org/javadoc/twitter4j/Query.html to customize the query
		System.out.println("********************************************************************************");
		System.out.println("***** TASK 1: search for all the tweets with the keywords: football world cup");
		
		String queryString = "football world cup";
		Query query = new Query(queryString);
		query.count(100); // sets the number of tweets to return per page, up to a max of 100
	    QueryResult result;
		try {
			result = twitter.search(query);
			Integer countTw = 1;
			System.out.println("Query result for " + queryString + ":");
			for (Status status : result.getTweets()) {
		        System.out.println(countTw++ + " > @" + status.getUser().getScreenName() + " (" + status.getCreatedAt().toString() + ") : " + status.getText() + "\n");
		    }
		} catch (TwitterException e) {
			logger.info("Exception while searching for tweets by a query string: " + e.getMessage());
			e.printStackTrace();
		}
	    
		
		// Task2: get all the tweets of a user (by user id) with paging control
		// The user ID of the NewYorkTimes Twitter account is: 807095l
		System.out.println("********************************************************************************");
		System.out.println("***** TASK 2: get all the tweets of a user (by user id) with paging control");
		
		Paging pagingInstance = new Paging();
		Integer pageNum = 1;
		Integer elementsPerPage = 40;
		pagingInstance.setPage(pageNum);
		pagingInstance.setCount(elementsPerPage);
		
		Long userId = 807095l;
		ResponseList<Status> timeline;
		try {
			timeline = twitter.getUserTimeline(userId, pagingInstance);
			if(timeline != null && timeline.size() > 0) {
				System.out.println("Retrieved " + timeline.size() + " tweets (user ID: "  + userId + ", page: " + (pageNum - 1) + ". Tweets per page: " + elementsPerPage + ")");
				Iterator<Status> statusIter = timeline.iterator();
				Integer countTw = 1;
				while(statusIter.hasNext()) {
					Status status = statusIter.next();
					if(status != null && status.getCreatedAt() != null) {
						System.out.println(countTw++ + " > @" + status.getUser().getScreenName() + " (" + status.getCreatedAt().toString() + ") : " + status.getText() + "\n");
					}
				}
			}
		} catch (TwitterException e) {
			logger.info("Exception while searching for tweets of a user timeline: " + e.getMessage());
			e.printStackTrace();
		}
		
	}

}
